import os
from pathlib import Path

from astromon.scripts import excluded_region

SKA = Path(os.environ["SKA"])


def modified_parser(parser_function):
    def get_parser():
        parser = parser_function()
        parser.description = "Periscope drift excluded regions management tool"
        # the default is db_file=None, which causes $SKA/data/astromon/astromon.h5 to be used
        parser.set_defaults(
            db_file=SKA / "data" / "periscope_drift_reports" / "excluded_sources.h5"
        )
        return parser

    return get_parser


def main(args=None):
    os.environ["ASTROMON_FILE"] = str(
        SKA / "data" / "periscope_drift_reports" / "excluded_sources.h5"
    )
    for action, parser in excluded_region.PARSERS.items():
        excluded_region.PARSERS[action] = modified_parser(parser)

    excluded_region.main(args)


if __name__ == "__main__":
    main()
