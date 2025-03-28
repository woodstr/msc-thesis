import os

def find_files(dir):
    '''Finds files in MAN directory'''
    files_name = os.listdir(dir)
    if 'info.csv' in files_name:
        files_name.remove('info.csv')

    # Get absolute paths
    files_abs = [os.path.abspath(os.path.join(dir, file)) for file in files_name]

    return files_name, files_abs

def check_filename_in_list(filename, filelist):
    '''Check if filename is in filelist strings'''
    for file in filelist:
        filename_ = file.split('_')[0]
        filename = filename.replace('##', '-')
        filename = filename.replace('#', '-')
        filename = filename.replace(' ', '-')
        if filename_ == filename:
            return True
    return False

if __name__ == '__main__':

    # Get raw files
    raw_files_name, raw_files_abs = find_files('../../../sem3/research_project/Research-Project-Data-Matrix-Code-/data/extracted/')

    # Get train/val/test split based on prev split
    train_files_name, train_files_abs = find_files('../../../sem3/research_project/Research-Project-Data-Matrix-Code-/data/MAN/images/train/')
    val_files_name, val_files_abs = find_files('../../../sem3/research_project/Research-Project-Data-Matrix-Code-/data/MAN/images/val/')
    test_files_name, test_files_abs = find_files('../../../sem3/research_project/Research-Project-Data-Matrix-Code-/data/MAN/images/test/')

    # Copy files to new directory
    for idx, filename in enumerate(raw_files_name):
        filename, extension = filename.split('.')[0], filename.split('.')[1]

        count = 0
        if check_filename_in_list(filename, train_files_name):
            # Copy file to new train dir
            os.system(f'copy "{raw_files_abs[idx]}" "C:\\Users\\aidan\\OneDrive\\Desktop\\itu\\msc\\courses\\sem4\\thesis\\msc-thesis\\data\\MAN\\raw\\train\\{filename}.{extension}"')
            count += 1

        if check_filename_in_list(filename, val_files_name):
            # Copy file to new val dir
            os.system(f'copy "{raw_files_abs[idx]}" "C:\\Users\\aidan\\OneDrive\\Desktop\\itu\\msc\\courses\\sem4\\thesis\\msc-thesis\\data\\MAN\\raw\\val\\{filename}.{extension}"')
            count += 1

        if check_filename_in_list(filename, test_files_name):
            # Copy file to new test dir
            os.system(f'copy "{raw_files_abs[idx]}" "C:\\Users\\aidan\\OneDrive\\Desktop\\itu\\msc\\courses\\sem4\\thesis\\msc-thesis\\data\\MAN\\raw\\test\\{filename}.{extension}"')
            count += 1

        if count != 1:
            print(filename, count)