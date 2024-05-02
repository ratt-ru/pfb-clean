# flake8: noqa
import click
from pfb import logo
logo()

@click.group()
def cli():
    pass


from pfb.workers import (init, grid, degrid, klean,
                         restore, spotless, model2comps,
                         fluxmop, fastim, smoovie)
