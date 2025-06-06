# illotka

Simulate the ill-conditioned Lotka-Volterra model.

## Dependencies

+ Python 3
+ NumPy
+ SciPy
+ Matplotlib

## Usage

+ A [Jupyter notebook](demos.ipynb) that demonstrates the use of the `illotka` package.

+ [Link to Google Colab notebook](https://colab.research.google.com/github/williamgilpin/illotka/blob/main/demos.ipynb)

+ [Link to static webpage](https://williamgilpin.github.io/illotka/demos.html)


## The ill-conditioned Lotka-Volterra model

We consider random ecosystems given by the generalized Lotka-Volterra equation,

$$
    \frac{dN_i}{dt} = N_i \left( r_i + \sum_{j=1}^N A_{ij} N_j \right)
$$

where $N_i$ is the population of species $i$, $r_i$ is the intrinsic growth rate of species $i$, and $A_{ij}$ is the interaction coefficient between species $i$ and $j$. The steady-state solutions of this equation has the form

$$
    -A \mathbf{N}^* = \mathbf{r}
$$

where $N_i \geq 0$ for all $i$.

In this notebook, we explore the behavior of this model when the interaction matrix $A_{ij}$, and the growth rates $r_i$ are drawn from random distributions. Specifically, we consider the case where $r_i \sim \mathcal{N}(0,1)$, and the matrix $A$ has the form

$$
    A = P^\top (Q - d\, I) P + \epsilon E
$$

where $Q_{ij} \sim \mathcal{N}(0,1)$, $E_{ij} \sim \mathcal{N}(0,1)$, $P$ is a low-rank matrix imposing functional redundancy, $d$ is a constant density-limitation, and $\epsilon \ll 1$ is a small constant.

## Reference

If you find this code or notebook useful, please consider citing the [paper](https://doi.org/10.1371/journal.pcbi.1013051) that describes the method.

```bibtex
@article{gilpin2025optimization,
    title={Optimization hardness constrains ecological transients},
    author={Gilpin, William},
    journal={PLOS Computational Biology},
    volume={21},
    number={5},
    pages={e1013051},
    year={2025},
    publisher={Public Library of Science San Francisco, CA USA}
}
```
