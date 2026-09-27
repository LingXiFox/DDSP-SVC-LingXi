import os
import yaml
import pickle
import torch


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_network_paras_amount(model_dict):
    info = dict()
    for model_name, model in model_dict.items():
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        info[model_name] = trainable_params
    return info


def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    return args


def convert_tensor_to_numpy(tensor, is_squeeze=True):
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


SPK_EMBED_KEY = 'ddsp_model.unit2ctrl.spk_embed.weight'


def expand_spk_embed_state(state, model, key=SPK_EMBED_KEY):
    """Migrate a checkpoint saved with n_spk=N into a model built with n_spk=N+K.

    Grows the speaker embedding tensor in `state` so load_state_dict does not
    raise a size mismatch. Old rows are copied verbatim (exact equality after
    load); new rows keep the model's own fresh initialization. Any OTHER shape
    mismatch is left untouched and still raises in load_state_dict - this
    helper never silently masks unrelated incompatibilities.

    Returns a (possibly shallow-copied) state dict; the input is not mutated.
    """
    if key not in state:
        return state
    model_state = model.state_dict()
    if key not in model_state:
        # model has n_spk == 1 (no spk embedding at all); leave the tensor as
        # an unexpected key for load_state_dict(strict=False) to ignore
        return state
    old = state[key]
    cur = model_state[key]
    if old.shape == cur.shape:
        return state
    if old.dim() != 2 or cur.dim() != 2 or old.shape[1] != cur.shape[1]:
        raise RuntimeError(
            f' [x] {key} shape mismatch cannot be migrated: '
            f'ckpt {tuple(old.shape)} vs model {tuple(cur.shape)}')
    if old.shape[0] > cur.shape[0]:
        raise RuntimeError(
            f' [x] checkpoint has MORE speakers ({old.shape[0]}) than the '
            f'model ({cur.shape[0]}); shrinking n_spk is not supported')
    expanded = cur.detach().clone()
    expanded[:old.shape[0]] = old.to(expanded.dtype)
    state = dict(state)
    state[key] = expanded
    print(f' [ckpt-compat] {key}: expanded {old.shape[0]} -> {cur.shape[0]} speakers; '
          f'rows 0..{old.shape[0]-1} copied verbatim from checkpoint, '
          f'rows {old.shape[0]}..{cur.shape[0]-1} keep fresh model initialization')
    return state


def optimizer_state_mismatch(optimizer, state):
    """Return None if `state` structurally matches `optimizer`, else a reason string.

    Compares param_group counts and per-group parameter counts, recursing into
    ChainedOptimizer sub-optimizer states. Detects layout changes (e.g.
    train.freeze_reflow removing reflow params from the optimizer) so callers
    can explicitly discard unmappable state instead of crashing mid-startup or
    pretending the optimizer was fully restored.
    """
    try:
        state_groups = state.get('param_groups')
        if state_groups is None:
            return 'checkpoint optimizer state has no param_groups'
        if len(state_groups) != len(optimizer.param_groups):
            return (f'param_groups count: ckpt {len(state_groups)} != '
                    f'current {len(optimizer.param_groups)}')
        for i, (sg, pg) in enumerate(zip(state_groups, optimizer.param_groups)):
            if len(sg.get('params', [])) != len(pg['params']):
                return (f'param_groups[{i}] size: ckpt {len(sg.get("params", []))} '
                        f'!= current {len(pg["params"])}')
        sub_states = state.get('optimizers')
        sub_opts = getattr(optimizer, 'optimizers', None)
        if sub_states is not None and sub_opts is not None:
            if len(sub_states) != len(sub_opts):
                return (f'chained sub-optimizer count: ckpt {len(sub_states)} '
                        f'!= current {len(sub_opts)}')
            for j, (ss, so) in enumerate(zip(sub_states, sub_opts)):
                reason = optimizer_state_mismatch(so, ss)
                if reason is not None:
                    return f'optimizers[{j}]: {reason}'
        return None
    except Exception as e:
        return f'unreadable optimizer state ({type(e).__name__}: {e})'


def load_model(
        expdir,
        model,
        optimizer,
        name='model',
        postfix='',
        device='cpu',
        load_optimizer=True):
    if postfix == '':
        postfix = '_' + postfix
    path = os.path.join(expdir, name+postfix)
    path_pt = traverse_dir(expdir, ['pt'], is_ext=False)
    global_step = 0
    if len(path_pt) > 0:
        steps = [s[len(path):] for s in path_pt]
        maxstep = max([int(s) if s.isdigit() else 0 for s in steps])
        if maxstep >= 0:
            path_pt = path+str(maxstep)+'.pt'
        else:
            path_pt = path+'best.pt'
        print(' [*] restoring model from', path_pt)
        ckpt = torch.load(path_pt, map_location=torch.device(device))
        global_step = ckpt['global_step']
        model_state = expand_spk_embed_state(ckpt['model'], model)
        model.load_state_dict(model_state, strict=False)
        if load_optimizer and ckpt.get('optimizer') != None:
            mismatch = optimizer_state_mismatch(optimizer, ckpt['optimizer'])
            if mismatch is None:
                optimizer.load_state_dict(ckpt['optimizer'])
                print(' [*] optimizer state: fully restored from checkpoint')
            else:
                # explicit discard, never silent (plan 17): model weights and
                # global_step above are still restored; only optimizer momentum
                # is lost and training continues with a fresh optimizer
                print(' [!] optimizer state: NOT restored, explicitly discarded.')
                print(' [!]   reason:', mismatch)
                print(' [!]   likely cause: parameter layout changed since the '
                      'checkpoint (e.g. train.freeze_reflow).')
                print(' [!]   continuing with a fresh optimizer; model weights '
                      'and global_step were restored.')
        elif load_optimizer:
            print(' [*] optimizer state: none in checkpoint (save_opt=false?), '
                  'starting with a fresh optimizer')
    return global_step, model, optimizer
