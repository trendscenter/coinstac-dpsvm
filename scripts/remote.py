import sys
import ujson as json

import numpy as np

from common_functions import list_recursive


def remote_0(args):
    input = args['input']
    # aggregate w_local and num_sample_local from local sites
    W_locals = np.array([
        site_dict['w_local'] for site, site_dict in input.items()
        if 'w_local' in site_dict
    ])

    num_sample_locals = np.array([
        site_dict['num_sample_local'] for site, site_dict in input.items()
        if 'num_sample_local' in site_dict
    ])

    output_dict = {'W_locals': W_locals.tolist(), 
                   'num_sample_locals': num_sample_locals,
                   'phase': 'remote_0'
    }

    # save a copy in cache
    result_dict = {'output': output_dict, 'cache': output_dict}
    return json.dumps(result_dict)


def remote_1(args):
    input = args['input']
    locals_dict = args['cache']
    # extract w_owner and num_sample_owner from the owner site
    for site, site_dict in input.items():
        if site_dict:
            owner_dict = site_dict
            break

    # combine w_owner and W_locals
    output_dict = {'confusion_matrix': owner_dict.get('confusion_matrix'),
                   'confusion_matrix_normalized': owner_dict.get('confusion_matrix_normalized'),
                   'w_owner': owner_dict.get('w_owner'),
                   'W_locals': locals_dict.get('W_locals'),
                   'num_sample_owner': owner_dict.get('num_sample_owner'),
                   'num_sample_locals': locals_dict.get('num_sample_locals')}

    result_dict = {'output': output_dict, 'success': True}
    return json.dumps(result_dict)


if __name__ == '__main__':
    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'phase'))

    if 'local_0' in phase_key:
        result_dict = remote_0(parsed_args)
        sys.stdout.write(result_dict)
    elif 'local_1' in phase_key:
        result_dict = remote_1(parsed_args)
        sys.stdout.write(result_dict)
    else:
        raise Exception('Error occurred at Remote')
