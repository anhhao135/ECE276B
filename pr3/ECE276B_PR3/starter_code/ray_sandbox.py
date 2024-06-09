import ray
ray.init(num_cpus=8, num_gpus=8, resources={"special_hardware": 1, "custom_label": 1})

@ray.remote
def f(x):
    return x * x

futures = [f.remote(i) for i in range(10000000)]
print(ray.get(futures)) # [0, 1, 4, 9]