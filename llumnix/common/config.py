# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import copy
from typing import Any
from ast import literal_eval
import yaml

from llumnix.logger import init_logger

logger = init_logger(__name__)

class Config(dict):
    """
    Config represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """
    def __init__(self, init_dict=None, key_list=None):
        """
        Args:
            init_dict (dict): the possibly-nested dictionary to initailize the Config.
            key_list (list[str]): a list of names which index this Config from the root.
                Currently only used for logging purposes.
            new_allowed (bool): whether adding new key is allowed when merging with
                other configs.
        """
        # Recursively convert nested dictionaries in init_dict into Configs
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        init_dict = self._create_config_tree_from_dict(init_dict, key_list)
        super(Config, self).__init__(init_dict)

    @classmethod
    def _create_config_tree_from_dict(cls, dic, key_list):
        """
        Create a configuration tree using the given dict.
        Any dict-like objects inside dict will be treated as a new Config.

        Args:
            dic (dict):
            key_list (list[str]): a list of names which index this Config from the root.
                Currently only used for logging purposes.
        """
        dic = copy.deepcopy(dic)
        for k, v in dic.items():
            if isinstance(v, dict):
                # Convert dict to Config
                dic[k] = cls(v, key_list=key_list + [k])        
        return dic
    
    @classmethod
    def _decode_cfg_value(cls, value):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.

        If the value is a dict, it will be interpreted as a new Config.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to Config objects
        if isinstance(value, dict):
            return cls(value)
        # All remaining processing is only applied to strings
        if not isinstance(value, str):
            return value
        # Try to interpret `value` as a:
        #   string, number, tuple, list, dict, boolean, or None
        try:
            value = literal_eval(value)
        # The following two excepts allow v to pass through when it represents a
        # string.
        #
        # Longer explanation:
        # The type of v is always a string (before calling literal_eval), but
        # sometimes it *represents* a string and other times a data structure, like
        # a list. In the case that v represents a string, what we got back from the
        # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
        # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
        # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
        # will raise a SyntaxError.
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value
    

    @staticmethod
    def load_yaml_with_base(filename: str, allow_unsafe: bool = False):
        """
        With "allow_unsafe=True", it supports pyyaml tags that evaluate
        expressions in config. See examples in
        https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
        Note that this may lead to arbitrary code execution: you must not
        load a config file from untrusted sources before manually inspecting
        the content of the file.
        Args:
            filename (str): the file name of the current config. Will be used to
                find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.
        Returns:
            (dict): the loaded yaml
        """
        with open(filename, "r", encoding='utf-8') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with open(filename, "r") as f:
                    cfg = yaml.unsafe_load(f)
        return cfg
    
    # def __str__(self):
    #     def _indent(s_, num_spaces):
    #         s = s_.split("\n")
    #         if len(s) == 1:
    #             return s_
    #         first = s.pop(0)
    #         s = [(num_spaces * " ") + line for line in s]
    #         s = "\n".join(s)
    #         s = first + "\n" + s
    #         return s

    #     r = ""
    #     s = []
    #     for k, v in sorted(self.items()):
    #         seperator = "\n" if isinstance(v, Config) else " "
    #         attr_str = "{}:{}{}".format(str(k), seperator, str(v))
    #         attr_str = _indent(attr_str, 2)
    #         s.append(attr_str)
    #     r += "\n".join(s)
    #     return r

    # def __repr__(self):
    #     return "{}({})".format(self.__class__.__name__, super(Config, self).__repr__())



    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False):
        """
        Merge configs from a given yaml file.
        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        loaded_cfg = self.load_yaml_with_base(
            cfg_filename, allow_unsafe=allow_unsafe
        )
        _merge_a_into_b(loaded_cfg, self, self, [])

    def clone(self):
        """Recursively copy this Config."""
        return copy.deepcopy(self)
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)
    
    def __setattr__(self, name, value):
        self[name] = value


def get_cfg() -> Config:
    """
    Get a copy of the default config.
    Returns:
        a Config instance.
    """
    from llumnix.common.defaults import _C

    return _C.clone()

def _merge_a_into_b(a, b, root, key_list):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if a is None:
        return
    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])

        v = copy.deepcopy(v_)
        v = b._decode_cfg_value(v)
        if k in b:
            # Recursively merge dicts
            if isinstance(v, Config):
                try:
                    _merge_a_into_b(v, b[k], root, key_list + [k])
                except BaseException:
                    raise
            else:
                b[k] = v
        else:
            raise KeyError("Non-existent config key: {}".format(full_key))
