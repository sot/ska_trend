#!/usr/bin/env python3
"""
Check continuity of centroid dashboard observation chains.

This module validates the prev/next linking structure in the centroid dashboard
reports by collecting all observations, ordering them by date, and walking the
chain backwards from the most recent observation to ensure all observations
are properly linked.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ska_helpers.logging import basic_logger

logger = basic_logger("check_continuity")


@dataclass
class ObservationInfo:
    """Container for observation information from info.json files."""

    obsid: int
    source: str
    date_starcat: str
    info_path: Path
    obs_links: Dict

    def __str__(self):
        return f"ObsID {self.obsid} ({self.source}) at {self.date_starcat}"

    def __repr__(self):
        return f"ObservationInfo(obsid={self.obsid}, source='{self.source}')"


@dataclass
class ContinuityChecker:
    """Check continuity of observation chains in centroid dashboard reports."""

    reports_root: Path
    observations: List[ObservationInfo] = field(default_factory=list)
    obs_by_key: Dict[Tuple[int, str], ObservationInfo] = field(default_factory=dict)

    def collect_observations(self) -> None:
        """Collect all observations from all years in reports directory."""
        if not self.reports_root.exists():
            logger.error(f"Reports directory not found: {self.reports_root}")
            return

        logger.info(f"Scanning observations in {self.reports_root}")

        # Process 4-digit year directories only
        for year_dir in sorted(self.reports_root.iterdir()):
            if not year_dir.is_dir():
                continue

            # Skip non-year directories - only process 4-digit years
            year_name = year_dir.name
            if not (year_name.isdigit() and len(year_name) == 4):
                logger.debug(f"Skipping non-4-digit-year directory: {year_name}")
                continue

            logger.info(f"Processing year {year_name}")

            for source_dir in sorted(year_dir.iterdir()):
                if not source_dir.is_dir():
                    continue

                logger.info(f"Checking {source_dir.name}")

                for obsid_dir in source_dir.iterdir():
                    if not obsid_dir.is_dir():
                        continue

                    info_json_path = obsid_dir / "info.json"
                    if not info_json_path.exists():
                        logger.debug(f"No info.json found in {obsid_dir}")
                        continue

                    try:
                        with open(info_json_path) as f:
                            info_data = json.load(f)

                        obs_info = ObservationInfo(
                            obsid=info_data["obsid"],
                            source=info_data["source"],
                            date_starcat=info_data["date_starcat"],
                            info_path=info_json_path,
                            obs_links=info_data["obs_links"],
                        )

                        self.observations.append(obs_info)
                        self.obs_by_key[(obs_info.obsid, obs_info.source)] = obs_info

                    except (KeyError, json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Error reading {info_json_path}: {e}")
                        continue

        logger.info(
            f"Collected {len(self.observations)} observations from 4-digit years"
        )

    def sort_observations_by_date(self) -> None:
        """Sort observations by date_starcat timestamp."""
        self.observations.sort(key=lambda obs: obs.date_starcat)
        logger.info("Sorted observations by date")

    def check_continuity(self) -> Tuple[bool, Optional[str]]:
        """
        Check continuity by walking backward from most recent observation.

        Returns:
            Tuple of (is_continuous, error_message)
        """
        if not self.observations:
            return False, "No observations found"

        # Start from the most recent observation
        current_obs = self.observations[-1]
        visited_keys = set()
        chain_count = 0

        logger.info(f"Starting continuity check from most recent: {current_obs}")

        while current_obs is not None:
            # Check for cycles
            obs_key = (current_obs.obsid, current_obs.source)
            if obs_key in visited_keys:
                return (
                    False,
                    f"Cycle detected at obsid {current_obs.obsid} (source {current_obs.source})",
                )

            visited_keys.add(obs_key)
            chain_count += 1

            # Get previous observation from links
            prev_link = current_obs.obs_links.get("prev")
            if prev_link is None:
                # Reached the beginning of the chain
                logger.info(f"Reached chain start at {current_obs}")
                break

            prev_obsid = prev_link["obsid"]
            prev_source = prev_link["source"]
            prev_key = (prev_obsid, prev_source)

            # Check if previous observation exists
            if prev_key not in self.obs_by_key:
                # Check if the obsid directory exists but lacks info.json in any year
                found_dir = None
                for year_dir in self.reports_root.iterdir():
                    if year_dir.is_dir() and year_dir.name.isdigit():
                        potential_dir = year_dir / prev_source / str(prev_obsid)
                        if potential_dir.exists():
                            found_dir = potential_dir
                            break

                if found_dir:
                    return False, (
                        f"Broken link: obsid {current_obs.obsid} (source {current_obs.source}) "
                        f"links to incomplete obsid {prev_obsid} (source {prev_source}) - "
                        f"directory exists but no info.json"
                    )
                else:
                    return False, (
                        f"Broken link: obsid {current_obs.obsid} (source {current_obs.source}) "
                        f"links to missing obsid {prev_obsid} (source {prev_source}) - "
                        f"directory does not exist"
                    )

            current_obs = self.obs_by_key[prev_key]

        # Check if all observations were visited
        if len(visited_keys) != len(self.observations):
            missing_count = len(self.observations) - len(visited_keys)
            missing_examples = [
                (obs.obsid, obs.source)
                for obs in self.observations
                if (obs.obsid, obs.source) not in visited_keys
            ][:5]  # Show first 5
            return False, (
                f"Incomplete chain: {missing_count} observations not "
                f"reachable from most recent. Examples: {missing_examples}"
            )

        logger.info(
            f"Chain complete: visited {chain_count} observations out of "
            f"{len(self.observations)} total"
        )
        return True, None

    def run_check(self) -> bool:
        """
        Run the complete continuity check.

        Returns:
            True if continuity is good, False if broken
        """
        logger.info("Starting centroid dashboard continuity check")

        self.collect_observations()
        if not self.observations:
            logger.error("No observations found to check")
            return False

        self.sort_observations_by_date()

        is_continuous, error_msg = self.check_continuity()

        if is_continuous:
            logger.info("✓ Observation chain continuity check PASSED")
            logger.info(
                f"Summary: {len(self.observations)} observations from 4-digit years, "
                f"all properly linked"
            )
        else:
            logger.error(f"✗ Observation chain continuity check FAILED: {error_msg}")
            logger.info(
                f"Summary: {len(self.observations)} observations from 4-digit years, "
                f"broken chain detected"
            )

        return is_continuous


def main():
    """Main entry point for continuity checker."""
    checker = ContinuityChecker(Path("reports"))
    success = checker.run_check()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
