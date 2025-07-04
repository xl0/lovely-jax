# 💘 Lovely JAX


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

[![](https://alexey.work/badge/)](https://alexey.work/?ref=lj-md)

## [Read full docs](https://xl0.github.io/lovely-jax) \| ❤️ [Lovely Tensors](https://github.com/xl0/lovely-tensors) \| 💟 [Lovely `NumPy`](https://github.com/xl0/lovely-numpy) \| [Discord](https://discord.gg/4NxRV7NH)

## Note: I’m pretty new to JAX

If something does not make sense, shoot me an
[Issue](https://github.com/xl0/lovely-jax/issues) or ping me on Discord
and let me know how it’s supposed to work!

Better support for sharded arrays and solid jit/pmap/vmap support coming
soon!

## Install

``` sh
pip install lovely-jax
```

## How to use

How often do you find yourself debugging JAX code? You dump an array to
the cell output, and see this:

``` python
numbers
```

    Array([[[-0.354, -0.337, -0.405, ..., -0.56 , -0.474,  2.249],
            [-0.405, -0.423, -0.491, ..., -0.919, -0.851,  2.163],
            [-0.474, -0.474, -0.542, ..., -1.039, -1.039,  2.198],
            ...,
            [-0.902, -0.834, -0.936, ..., -1.467, -1.296,  2.232],
            [-0.851, -0.782, -0.936, ..., -1.604, -1.501,  2.18 ],
            [-0.834, -0.816, -0.971, ..., -1.656, -1.553,  2.112]],

           [[-0.197, -0.197, -0.303, ..., -0.478, -0.373,  2.411],
            [-0.25 , -0.232, -0.338, ..., -0.705, -0.67 ,  2.359],
            [-0.303, -0.285, -0.39 , ..., -0.74 , -0.81 ,  2.376],
            ...,
            [-0.425, -0.232, -0.373, ..., -1.09 , -1.02 ,  2.429],
            [-0.39 , -0.232, -0.425, ..., -1.23 , -1.23 ,  2.411],
            [-0.408, -0.285, -0.478, ..., -1.283, -1.283,  2.341]],

           [[-0.672, -0.985, -0.881, ..., -0.968, -0.689,  2.396],
            [-0.724, -1.072, -0.968, ..., -1.247, -1.02 ,  2.326],
            [-0.828, -1.125, -1.02 , ..., -1.264, -1.16 ,  2.379],
            ...,
            [-1.229, -1.473, -1.386, ..., -1.508, -1.264,  2.518],
            [-1.194, -1.456, -1.421, ..., -1.648, -1.473,  2.431],
            [-1.229, -1.526, -1.508, ..., -1.682, -1.526,  2.361]]], dtype=float32)

Was it really useful for you, as a human, to see all these numbers?

What is the shape? The size?  
What are the statistics?  
Are any of the values `nan` or `inf`?  
Is it an image of a man holding a tench?

``` python
import lovely_jax as lj
```

``` python
lj.monkey_patch()
```

## Summary

``` python
numbers
```

    Array[196, 196, 3] n=115248 (0.4Mb) x∈[-2.118, 2.640] μ=-0.388 σ=1.073 cpu:0

Better, huh?

``` python
numbers[1,:6,1] # Still shows values if there are not too many.
```

    Array[6] x∈[-0.408, -0.232] μ=-0.340 σ=0.075 cpu:0 [-0.250, -0.232, -0.338, -0.408, -0.408, -0.408]

``` python
spicy = numbers.flatten()[:12].copy()

spicy = (spicy  .at[0].mul(10000)
                .at[1].divide(10000)
                .at[2].set(float('inf'))
                .at[3].set(float('-inf'))
                .at[4].set(float('nan'))
                .reshape((2,6)))
spicy # Spicy stuff
```

    Array[2, 6] n=12 x∈[-3.541e+03, -1.975e-05] μ=-393.848 σ=1.113e+03 +Inf! -Inf! NaN! cpu:0

``` python
jnp.zeros((10, 10)) # A zero array - make it obvious
```

    Array[10, 10] n=100 all_zeros cpu:0

``` python
spicy.v # Verbose
```

    Array[2, 6] n=12 x∈[-3.541e+03, -1.975e-05] μ=-393.848 σ=1.113e+03 +Inf! -Inf! NaN! cpu:0
    Array([[-3.541e+03, -1.975e-05,        inf,       -inf,        nan, -9.853e-01],
           [-4.054e-01, -3.025e-01, -8.807e-01, -4.397e-01, -3.025e-01, -7.761e-01]], dtype=float32)

``` python
spicy.p # The plain old way
```

    Array([[-3.541e+03, -1.975e-05,        inf,       -inf,        nan, -9.853e-01],
           [-4.054e-01, -3.025e-01, -8.807e-01, -4.397e-01, -3.025e-01, -7.761e-01]], dtype=float32)

## Going `.deeper`

``` python
numbers.deeper
```

    Array[196, 196, 3] n=115248 (0.4Mb) x∈[-2.118, 2.640] μ=-0.388 σ=1.073 cpu:0
      Array[196, 3] n=588 x∈[-1.912, 2.411] μ=-0.728 σ=0.519 cpu:0
      Array[196, 3] n=588 x∈[-1.861, 2.359] μ=-0.778 σ=0.450 cpu:0
      Array[196, 3] n=588 x∈[-1.758, 2.379] μ=-0.838 σ=0.437 cpu:0
      Array[196, 3] n=588 x∈[-1.656, 2.466] μ=-0.878 σ=0.415 cpu:0
      Array[196, 3] n=588 x∈[-1.717, 2.448] μ=-0.882 σ=0.399 cpu:0
      Array[196, 3] n=588 x∈[-1.717, 2.431] μ=-0.905 σ=0.408 cpu:0
      Array[196, 3] n=588 x∈[-1.563, 2.448] μ=-0.859 σ=0.416 cpu:0
      Array[196, 3] n=588 x∈[-1.475, 2.431] μ=-0.791 σ=0.463 cpu:0
      Array[196, 3] n=588 x∈[-1.526, 2.429] μ=-0.759 σ=0.499 cpu:0
      ...

``` python
# You can go deeper if you need to
numbers[:3,:5,:3].deeper(2)
```

    Array[3, 5, 3] n=45 x∈[-1.316, -0.197] μ=-0.593 σ=0.302 cpu:0
      Array[5, 3] n=15 x∈[-0.985, -0.197] μ=-0.491 σ=0.267 cpu:0
        Array[3] x∈[-0.672, -0.197] μ=-0.408 σ=0.197 cpu:0 [-0.354, -0.197, -0.672]
        Array[3] x∈[-0.985, -0.197] μ=-0.507 σ=0.343 cpu:0 [-0.337, -0.197, -0.985]
        Array[3] x∈[-0.881, -0.303] μ=-0.530 σ=0.252 cpu:0 [-0.405, -0.303, -0.881]
        Array[3] x∈[-0.776, -0.303] μ=-0.506 σ=0.199 cpu:0 [-0.440, -0.303, -0.776]
        Array[3] x∈[-0.916, -0.215] μ=-0.506 σ=0.298 cpu:0 [-0.388, -0.215, -0.916]
      Array[5, 3] n=15 x∈[-1.212, -0.232] μ=-0.609 σ=0.302 cpu:0
        Array[3] x∈[-0.724, -0.250] μ=-0.460 σ=0.197 cpu:0 [-0.405, -0.250, -0.724]
        Array[3] x∈[-1.072, -0.232] μ=-0.576 σ=0.360 cpu:0 [-0.423, -0.232, -1.072]
        Array[3] x∈[-0.968, -0.338] μ=-0.599 σ=0.268 cpu:0 [-0.491, -0.338, -0.968]
        Array[3] x∈[-0.968, -0.408] μ=-0.651 σ=0.235 cpu:0 [-0.577, -0.408, -0.968]
        Array[3] x∈[-1.212, -0.408] μ=-0.761 σ=0.336 cpu:0 [-0.662, -0.408, -1.212]
      Array[5, 3] n=15 x∈[-1.316, -0.285] μ=-0.677 σ=0.306 cpu:0
        Array[3] x∈[-0.828, -0.303] μ=-0.535 σ=0.219 cpu:0 [-0.474, -0.303, -0.828]
        Array[3] x∈[-1.125, -0.285] μ=-0.628 σ=0.360 cpu:0 [-0.474, -0.285, -1.125]
        Array[3] x∈[-1.020, -0.390] μ=-0.651 σ=0.268 cpu:0 [-0.542, -0.390, -1.020]
        Array[3] x∈[-1.003, -0.478] μ=-0.708 σ=0.219 cpu:0 [-0.645, -0.478, -1.003]
        Array[3] x∈[-1.316, -0.513] μ=-0.865 σ=0.336 cpu:0 [-0.765, -0.513, -1.316]

## Now in `.rgb` color

The important queston - is it our man?

``` python
numbers.rgb
```

![](index_files/figure-commonmark/cell-13-output-1.png)

*Maaaaybe?* Looks like someone normalized him.

``` python
in_stats = ( (0.485, 0.456, 0.406),     # mean
             (0.229, 0.224, 0.225) )    # std

# numbers.rgb(in_stats, cl=True) # For channel-last input format
numbers.rgb(in_stats)
```

![](index_files/figure-commonmark/cell-14-output-1.png)

It’s indeed our hero, the Tenchman!

## `.plt` the statistics

``` python
(numbers+3).plt
```

![](index_files/figure-commonmark/cell-15-output-1.svg)

``` python
(numbers+3).plt(center="mean", max_s=1000)
```

![](index_files/figure-commonmark/cell-16-output-1.svg)

``` python
(numbers+3).plt(center="range")
```

![](index_files/figure-commonmark/cell-17-output-1.svg)

## See the `.chans`

``` python
# .chans will map values betwen [-1,1] to colors.
# Make our values fit into that range to avoid clipping.
mean = jnp.array(in_stats[0])
std = jnp.array(in_stats[1])
numbers_01 = (numbers*std + mean)
numbers_01
```

    Array[196, 196, 3] n=115248 (0.4Mb) x∈[0., 1.000] μ=0.361 σ=0.248 cpu:0

``` python
numbers_01.chans
```

![](index_files/figure-commonmark/cell-19-output-1.png)

## Grouping

``` python
# Make 8 images with progressively higher brightness and stack them 2x2x2.
eight_images = (jnp.stack([numbers]*8) + jnp.linspace(-2, 2, 8)[:,None,None,None])
eight_images = (eight_images
                     *jnp.array(in_stats[1])
                     +jnp.array(in_stats[0])
                ).clip(0,1).reshape(2,2,2,196,196,3)

eight_images
```

    Array[2, 2, 2, 196, 196, 3] n=921984 (3.5Mb) x∈[0., 1.000] μ=0.382 σ=0.319 cpu:0

``` python
eight_images.rgb
```

![](index_files/figure-commonmark/cell-21-output-1.png)

## Sharding

``` python
assert jax.__version_info__[0] == 0
if jax.__version_info__[1] >= 4:
    from jax.sharding import PositionalSharding
    from jax.experimental import mesh_utils
    sharding = PositionalSharding(mesh_utils.create_device_mesh((4,2)))
    x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
    y = jax.device_put(x, sharding)

    jax.debug.visualize_array_sharding(y)
else:
    # Note: Looks like ShardedDeviceArray needs an explicit device axis?
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 1024, 8192))
    y = jax.device_put_sharded([x for x in x], jax.devices())

print(x)
print(y)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">            </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">   CPU 0    </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">   CPU 1    </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #393b79">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #d6616b">            </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">            </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">   CPU 2    </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">   CPU 3    </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8ca252">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #de9ed6">            </span>
<span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">            </span>
<span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">   CPU 4    </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">   CPU 5    </span>
<span style="color: #000000; text-decoration-color: #000000; background-color: #e7cb94">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #6b6ecf">            </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">            </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">   CPU 6    </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">   CPU 7    </span>
<span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #a55194">            </span><span style="color: #ffffff; text-decoration-color: #ffffff; background-color: #8c6d31">            </span>
</pre>

    Array[8192, 8192] n=67108864 (0.2Gb) x∈[-5.420, 5.220] μ=-0.000 σ=1.000 cpu:0
    Array[8192, 8192] n=67108864 (0.2Gb) x∈[-5.420, 5.220] μ=-0.000 σ=1.000 cpu:0,1,2,3,4,5,6,7

## Options \| [Docs](utils.config.html)

``` python
from lovely_jax import set_config, config, lovely, get_config
```

``` python
set_config(precision=5, sci_mode=True, color=False)
jnp.array([1., 2, jnp.nan])
```

    Array[3] μ=1.50000e+00 σ=5.00000e-01 NaN! cpu:0 [1.00000e+00, 2.00000e+00, nan]

``` python
set_config(precision=None, sci_mode=None, color=None) # None -> Reset to defaults
```

``` python
print(jnp.array([1., 2]))
# Or with config context manager.
with config(sci_mode=True, precision=5):
    print(jnp.array([1., 2]))

print(jnp.array([1., 2]))
```

    Array[2] μ=1.500 σ=0.500 cpu:0 [1.000, 2.000]
    Array[2] μ=1.50000e+00 σ=5.00000e-01 cpu:0 [1.00000e+00, 2.00000e+00]
    Array[2] μ=1.500 σ=0.500 cpu:0 [1.000, 2.000]

## Without `.monkey_patch`

``` python
lj.lovely(spicy)
```

    Array[2, 6] n=12 x∈[-3.541e+03, -1.975e-05] μ=-393.848 σ=1.113e+03 +Inf! -Inf! NaN! cpu:0

``` python
lj.lovely(spicy, verbose=True)
```

    Array[2, 6] n=12 x∈[-3.541e+03, -1.975e-05] μ=-393.848 σ=1.113e+03 +Inf! -Inf! NaN! cpu:0
    Array([[-3.541e+03, -1.975e-05,        inf,       -inf,        nan, -9.853e-01],
           [-4.054e-01, -3.025e-01, -8.807e-01, -4.397e-01, -3.025e-01, -7.761e-01]], dtype=float32)

``` python
lj.lovely(numbers, depth=1)
```

    Array[196, 196, 3] n=115248 (0.4Mb) x∈[-2.118, 2.640] μ=-0.388 σ=1.073 cpu:0
      Array[196, 3] n=588 x∈[-1.912, 2.411] μ=-0.728 σ=0.519 cpu:0
      Array[196, 3] n=588 x∈[-1.861, 2.359] μ=-0.778 σ=0.450 cpu:0
      Array[196, 3] n=588 x∈[-1.758, 2.379] μ=-0.838 σ=0.437 cpu:0
      Array[196, 3] n=588 x∈[-1.656, 2.466] μ=-0.878 σ=0.415 cpu:0
      Array[196, 3] n=588 x∈[-1.717, 2.448] μ=-0.882 σ=0.399 cpu:0
      Array[196, 3] n=588 x∈[-1.717, 2.431] μ=-0.905 σ=0.408 cpu:0
      Array[196, 3] n=588 x∈[-1.563, 2.448] μ=-0.859 σ=0.416 cpu:0
      Array[196, 3] n=588 x∈[-1.475, 2.431] μ=-0.791 σ=0.463 cpu:0
      Array[196, 3] n=588 x∈[-1.526, 2.429] μ=-0.759 σ=0.499 cpu:0
      ...

``` python
lj.rgb(numbers, in_stats)
```

![](index_files/figure-commonmark/cell-30-output-1.png)

``` python
lj.plot(numbers, center="mean")
```

![](index_files/figure-commonmark/cell-31-output-1.svg)

``` python
lj.chans(numbers_01)
```

![](index_files/figure-commonmark/cell-32-output-1.png)

## Matplotlib integration \| [Docs](matplotlib.html)

``` python
numbers.rgb(in_stats).fig # matplotlib figure
```

![](index_files/figure-commonmark/cell-33-output-1.png)

``` python
(numbers*0.3+0.5).chans.fig # matplotlib figure
```

![](index_files/figure-commonmark/cell-34-output-1.png)

``` python
numbers.plt.fig.savefig('pretty.svg') # Save it
```

``` python
!file pretty.svg; rm pretty.svg
```

    pretty.svg: SVG Scalable Vector Graphics image

### Add content to existing Axes

``` python
fig = plt.figure(figsize=(8,3))
fig.set_constrained_layout(True)
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1,1:])

ax2.set_axis_off()
ax3.set_axis_off()

numbers_01.plt(ax=ax1)
numbers_01.rgb(ax=ax2)
numbers_01.chans(ax=ax3);
```

![](index_files/figure-commonmark/cell-37-output-1.png)
