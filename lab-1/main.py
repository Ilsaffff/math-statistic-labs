import numpy as np
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
from math import gamma


def plot_normal_distribution(sizes: tuple = (10, 50, 100)):
    grid = np.linspace(-5, 5, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r'Случайная величина $\xi \sim \mathcal{N}(0, 1)$')

    for i in range(len(sizes)):
        sp_normal_distribution = np.random.standard_normal(size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(sp_normal_distribution, bins=30, density=True,
                 alpha=0.6, label='Гистограмма выборки')
        plt.plot(grid, sps.norm.pdf(grid), color='red',
                 lw=5, label='Плотность случайной величины')
        plt.title(f'\nРазмер выборки: {sizes[i]}', fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.show()


def plot_cauchy_distribution(sizes: tuple = (10, 50, 100)):
    grid = np.linspace(-30, 30, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r'Случайная величина $\xi \sim \mathcal{C}(0, 1)$')

    for i in range(len(sizes)):
        cauchy_distr = sps.cauchy.rvs(loc=0, scale=1, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.xlim([-15, 15])
        sns.histplot(cauchy_distr, kde=False, stat='density', label='samples')
        plt.plot(grid, sps.cauchy.pdf(grid), color='red',
                 lw=3, label='Плотность случайной величины')
        plt.title(f'\nРазмер выборки: {sizes[i]}', fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.show()


def plot_student_t_distribution(df: int = 3, sizes: tuple = (10, 50, 100)):
    grid = np.linspace(-5, 5, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r'Случайная величина $\xi \sim t(0, 3)$')

    for i in range(len(sizes)):
        student_distr = np.random.standard_t(df, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(student_distr, bins=30, density=True,
                 alpha=0.6, label='Гистограмма выборки')
        y = sps.t.pdf(grid, df)
        plt.plot(grid, y, color='red',
                 lw=3, label='Плотность случайной величины')
        plt.title(f'\nРазмер выборки: {sizes[i]}', fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.show()


def plot_poisson_distribution(sizes: tuple = (10, 50, 100)):
    grid = np.linspace(0, 20, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r'Случайная величина $\xi \sim \mathcal{P}(10)$')

    for i in range(len(sizes)):
        cauchy_distr = np.random.poisson(lam=10, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(cauchy_distr, bins=30, density=True,
                 alpha=0.6, label='Гистограмма выборки')
        y = [(10 ** x * np.exp(-10) / gamma(x + 1)) for x in grid]
        plt.plot(grid, y, color='red',
                 lw=3, label='Плотность случайной величины')
        plt.title(f'\nРазмер выборки: {sizes[i]}', fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.show()


def plot_uniform_distribution(sizes: tuple = (10, 50, 100)):
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(
        r'Случайная величина $\xi \sim \mathcal{U}(-\sqrt{3}, \sqrt{3})$')

    for i in range(len(sizes)):
        cauchy_distr = np.random.uniform(low=-np.sqrt(4.0), high=np.sqrt(4.0),
                                         size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(cauchy_distr, bins=30, density=True,
                 alpha=0.6, label='Гистограмма выборки')
        plt.plot(grid, sps.uniform.pdf(grid, loc=-np.sqrt(3.0),
                                       scale=2 * np.sqrt(3.0)), color='red',
                 lw=3, label='Плотность случайной величины')
        plt.title(f'\nРазмер выборки: {sizes[i]}', fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.show()


if __name__ == '__main__':
    plot_normal_distribution()
    plot_cauchy_distribution()
    plot_student_t_distribution()
    plot_poisson_distribution()
    plot_uniform_distribution()