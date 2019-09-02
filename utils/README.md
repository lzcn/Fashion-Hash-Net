# Package `utils`

This package include:

- `check`: file and folder checking.
- `tracer`: plot utils for training, based on `visdom`.

## `check` module

`check` is a module for checking whether directory exists that support the following methods:

- `check_dirs(folders, action='check', mode='all', verbose=False):`

- Parameters:

  - `folders`: single folder or a list of folders to check
  - `action`: the action will take after checking. The default action is 'check' for only checking the existence. Another choice is 'mkdir' which will make new folder when it doesn't exits.
  - `mode`: whether it requires all folders exits or any one exits in `folders`

- Return:

  - `bool`: return the the final decision according to setting.

- `check_files(files, mode='any', verbose=False):`

- Parameters:

  - `files`: single file or a list of files to check
  - `mode`: whether it requires all folders exits or any one exits in `files`

- `list_files(folder, suffix='', recursive=False):`

- Parameters:

  - `folder`: the folder to process
  - `suffix`: file suffix. It can be a list of suffixes
  - `recursive`: whether to process sub-folder

- `check_exists(lists, mode='any', verbose=False):`

- Parameters

  - `lists`: a list of files and folders to check. Only the existence will be check.
