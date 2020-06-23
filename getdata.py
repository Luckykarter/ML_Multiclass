import os
import zipfile
import easygui


def _unzip_all(dir):
    for path in os.listdir(dir):
        s_path = os.path.splitext(path)
        if s_path[1].lower() == '.zip':
            local_zip = os.path.join(dir, path)
            zip_ref = zipfile.ZipFile(local_zip, 'r')
            new_dir = os.path.join(dir, s_path[0])
            zip_ref.extractall(new_dir)
            zip_ref.close()


# load images of humans and horses from ZIP-file
def get_data():
    # add generalized directory
    train_dir = None
    while train_dir is None:
        train_dir = easygui.diropenbox(title='Choose folder that contains training set with types split in folders')
        if len(os.listdir(train_dir)) < 2 or train_dir is None:
            if easygui.ynbox('Directory must contain at least two sub-directories.\nChoose another directory?'):
                train_dir = None
            else:
                exit()
    dirs = os.listdir(train_dir)

    # validation data is not mandatory
    while True:
        validation_dir = easygui.diropenbox(title='Choose directory that contains validation data of same types')
        if not validation_dir:
            break
        if os.listdir(validation_dir) != dirs:
            if not easygui.ynbox('Validation directory must contain same sub-directories as training set\nChoose '
                                 'another directory?'):
                validation_dir = None
                break
        else:
            break

    # if directory is chosen - unzipping is not needed
    # if not unzipped:
    #     unZipAll(base_dir)

    # train_dir = os.path.join(base_dir, name_dir)
    # validation_dir = os.path.join(base_dir, 'validation-' + name_dir)

    train_dirs = [os.path.join(train_dir, x)
                  for x in os.listdir(train_dir)]

    validation_dirs = []
    if validation_dir is not None:
        validation_dirs = [os.path.join(validation_dir, x)
                           for x in os.listdir(validation_dir)]

    labels = [os.path.basename(x).capitalize()
              for x in train_dirs]
    print(labels)
    for item in ((train_dirs, 'training'), (validation_dirs, 'validation')):
        dirs = item[0]
        name = item[1]
        for i in range(0, len(dirs)):
            print('Total {} {} images: {}'.format(
                name, labels[i], len(os.listdir(dirs[i]))
            ))

    labels.sort()
    return labels, train_dirs, validation_dirs

def get_user_file(labels):

     return easygui.fileopenbox(
        msg="Types: {}".format(', '.join(labels)),
        title="Choose image(s)",
        multiple=True
    )