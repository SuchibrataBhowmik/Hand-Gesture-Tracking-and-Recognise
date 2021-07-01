class IncompatibleFolderStructure(Exception):
    def __init__(self, msg=None):
        info = ("\n"
                "The given coustom Dataset root directory folder structure does not correct. "
                "Maintain the following structure:\n\n"
                "hand_dataset/\n"
                "├── test\n"
                "│   └── different scenes\n"
                "|       └── all frames\n"
                "├── train\n"
                "|   └── different scene\n"
                "|       ├── annotations\n"
                "|       |   └── all frames annotations\n"
                "|       └── data\n"
                "|           └── all frames\n"
                "└── validation\n"
                "    └── different scene\n"
                "        ├── annotations\n"
                "        |   └── all frames annotations\n"
                "        └── data\n"
                "            └── all frames\n")        
        if msg is not None:
            info = info + msg
        super().__init__(info)


class InvalidOption(Exception):
    pass

class IncompleteArgument(Exception):
    pass
