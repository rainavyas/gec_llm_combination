import os

def next_dir(path, dir_name, create=True):
    if not os.path.isdir(f'{path}/{dir_name}'):
        if create:
            os.mkdir(f'{path}/{dir_name}')
        else:
            raise ValueError ("provided args do not give a valid model path")
    path += f'/{dir_name}'
    return path