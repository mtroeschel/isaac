import asyncio
import logging

import aiomas
import click

from openvpp_agents import util


@click.command()
@click.option('--start-date', required=True,
              callback=util.validate_start_date,
              help='Start date for the simulation (ISO-8601 compliant, e.g.: '
                   '2010-03-27T00:00:00+01:00')
@click.option('--log-level', '-l', default='info', show_default=True,
              type=click.Choice(['debug', 'info', 'warning', 'error',
                                 'critical']),
              help='Log level for the MAS')
@click.argument('addr', metavar='HOST:PORT', callback=util.validate_addr)
def main(addr, start_date, log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))
    container_kwargs = util.get_container_kwargs(start_date)
    try:
        aiomas.run(aiomas.subproc.start(addr, **container_kwargs))
    finally:
        asyncio.get_event_loop().close()


if __name__ == '__main__':
    main()
