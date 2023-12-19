import configparser
import uuid

config_default = "subl.conf.default"
config = configparser.ConfigParser()
config_filename = None


def parse_config(filename="subl.conf"):
    global config_filename
    config_filename = filename

    config.sections()
    config.read(filename)
    if not config.has_option(None, "host"):
        config.read(config_default)
        update_config()


def update_config():
    with open(config_filename, "w") as configfile:
        config.write(configfile)
