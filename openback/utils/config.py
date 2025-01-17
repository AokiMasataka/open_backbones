import json

def _load_config(config: dict, key: str) -> dict:
    if isinstance(config[key], str):
        with open(config[key], 'r') as f:
            config[key] = json.load(f)[key]
    
    return config


def load_config_file(path: str) -> dict:
    with open(path, 'r') as f:
        config = json.load(fp=f)

    config = _load_config(config=config, key='backbone')
    
    return config
    

def dump_config_file(config: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(obj=config, fp=f)


def config_encoder(config: dict, indent: int = 0) -> str:
    text = ''
    for key, value in config.items():
        if isinstance(value, dict):
            text += '\t' * indent + f'{key}=' + 'dict(' + '\n'
            indent += 1
            text += config_encoder(config=value, indent=indent)
            indent -= 1
            text += '\t' * indent + ')' + ',\n'
        elif isinstance(value, list):
            text += '\t' * indent + f'{key}=' + '[' + '\n'
            indent += 1
            for item in value:
                indent += 1
                text += '\t' * indent + str(item) + ',\n'
                indent -= 1
            indent -= 1
            text += '\t' * indent + ']' + ',\n'

        else:
            if isinstance(value, str):
                value = "'" + value + "'"
            else:
                value = str(value)
            text += '\t' * indent + f'{key}=' + value + ',\n'
    return text