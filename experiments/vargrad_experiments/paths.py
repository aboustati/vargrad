import os


def config_path():
    path = os.path.realpath(__file__)
    path = os.path.join('/', *path.split('/')[:7])
    path = os.path.join(path, 'configs')
    return path


def results_path():
    path = os.path.realpath(__file__).split('/')[:-2]
    path = os.path.join('/', *path, 'results')
    return path


CONFIG_PATH = config_path()
RESULTS_PATH = results_path()


if __name__ == '__main__':
    print('Config path:', CONFIG_PATH)
    print('Results path:', RESULTS_PATH)