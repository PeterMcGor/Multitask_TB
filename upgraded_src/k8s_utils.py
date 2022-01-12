import os
import itertools
import collections
import yaml
from datetime import datetime

def iterate_nested_yaml(yaml_params, key, out_e = []):
    """
    :param yaml_params: dict from yaml.load function
    :param key: Looking for
    :return:
    """
    if isinstance(yaml_params, dict):
        if key in yaml_params:
            return key, out_e
        out = [k for k, value in yaml_params.items() if isinstance(value, str)]
        [yaml_params.pop(k) for k in out]

        for k in yaml_params:
            out_e.append(k)
            params = yaml_params[k]
            res = iterate_nested_yaml(params, key, out_e=out_e)
            if res is not None and res[0] == key:
                return key, out_e
            out_e = []
    else:
        if isinstance(yaml_params, collections.Iterable):
            for i, element in enumerate(yaml_params):
                if not isinstance(element, str):
                    out_e.append(i)
                    res = iterate_nested_yaml(element, key, out_e=out_e)
                    if res is not None and res[0] == key:
                        return key, out_e
                    out_e = []


def modify_yml_file(yml_file, dict_params_to_modify, outfile=None, default_flow_style=False):
    """

    :param yml_file:
    :param dict_params_to_modify:
    :param outfile:
    :return:
    """
    import copy
    with open(yml_file, 'r') as stream:
        params = yaml.load(stream)
    for k, value in dict_params_to_modify.items():
        iter_out = iterate_nested_yaml(copy.deepcopy(params), k, out_e=[])
        if iter_out is not None:
            route = iter_out[1]
            if len(route) > 0:
                to_change = params[route[0]]
                # its supouses that iterate_nested_yaml just going to return keys from yaml dict or list indices
                for node in route[1:]:
                    to_change = to_change[node]
                to_change[k] = value
            else:
                params[k] = value

    outfile = yml_file if outfile is None else outfile
    with open(outfile, 'w') as writer:
        yaml.dump(params, writer, default_flow_style=default_flow_style,  allow_unicode=True)


def gen_hyperparameters_optimization_yamls(caller_yml_template, config_yml_template, hyper_params_dict,
                                           args_list=["-u", "src/task_laucher.py"],
                                           experiment_name_field="experiment", calling_ymls_path="../callings_yml/",
                                           configs_ymls_path="../configs_yml/"):
    print("File", os.path.dirname(os.path.abspath(__file__)))

    keys, values = zip(*hyper_params_dict.items())
    hyperparam_allsets = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print("Total number of hyperparameter sets: " + str(len(hyperparam_allsets)))

    for hyper_set in hyperparam_allsets:
        job_id = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
        experiment_name = "experiment-"+job_id
        experiment_file = job_id+'.yml'
        hyper_set[experiment_name_field] = experiment_name
        config_file_path = os.path.join(configs_ymls_path, experiment_file)
        modify_yml_file(config_yml_template, hyper_set, outfile=config_file_path, default_flow_style=False)
        modify_yml_file(caller_yml_template, {"args": args_list+["-c", os.path.join("configs_yml", experiment_file)],
                                              "name": experiment_name},
                        outfile=os.path.join(calling_ymls_path, "calling_"+experiment_file))


if __name__ == '__main__':
    print('Hello')
    yml_file = "/home/pmacias/Projects/bodyct-tuberculosis-multitask/callings_yml/general_yml_1.yml"
    config_yml_template = "/home/pmacias/Projects/bodyct-tuberculosis-multitask/configs_yml_templates/config_template_1.yml"
    calling_yml_template = "/home/pmacias/Projects/bodyct-tuberculosis-multitask/callings_yml_templates/general_yml_1.yml"
    # modify_yml_file("/home/pmacias/conf.yml", {"optimizer": "None", "optimizer_params": [0.69, 42, 1e-4]})
    # modify_yml_file("/home/pmacias/gen.yml", {"args":"None none"}, outfile="/home/pmacias/gen_mod.yml")
    # with open(yml_file, 'r') as stream:
    #     params = yaml.load(stream)
    # output = iterate_nested_yaml(params.copy(), "args")
    # print("params", params)
    # if output is not None:
    #     z = output[0]
    #     route = output[1]
    #     print("Z", z, "route", route)
    #     par = params
    #     for node in route:
    #        par = par[node]
    #     print(par[z])
    #     par[z] = "tus muertos"
    #     print(par)
    #     print(params)
    # modify_yml_file("/home/pmacias/Projects/bodyct-tuberculosis-multitask/callings_yml/general_yml_2.yml", {"args": [1,2,3,"4"], "restartPolicy": "tu madre"}, outfile="/tmp/dumper3.yml")
    gen_hyperparameters_optimization_yamls(calling_yml_template, config_yml_template, {"num_channels": [8, 16], "activation": ["relu", "prelu"]})



