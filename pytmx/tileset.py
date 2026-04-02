"""
Copyright (C) 2012-2025, Leif Theden <leif.theden@gmail.com>

This file is part of pytmx.

pytmx is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

pytmx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with pytmx.  If not, see <https://www.gnu.org/licenses/>.

Tiled Tileset parser and model.
"""

import logging
import os
from typing import TYPE_CHECKING, Any

try:  # Python 3.11+
    from typing import Self
except Exception:  # Python < 3.11
    from typing_extensions import Self

from xml.etree import ElementTree
from xml.etree.ElementTree import ParseError

from .constants import AnimationFrame
from .element import TiledElement
from .properties import parse_properties, types

if TYPE_CHECKING:
    from .map import TiledMap

logger = logging.getLogger(__name__)


class TiledTileset(TiledElement):
    """Represents a Tiled Tileset

    External tilesets are supported.  GID/ID's from Tiled are not
    guaranteed to be the same after loaded.
    """

    def __init__(self, parent: "TiledMap", node: ElementTree.Element) -> None:
        """Represents a Tiled Tileset

        Args:
            parent (TiledMap): The parent TiledMap.
            node (ElementTree.Element): The XML node.
        """
        super().__init__()
        self.parent = parent
        self.offset: tuple[int, int] = (0, 0)
        self.tileset_source: str | None = None

        # defaults from the specification
        self.firstgid: int = 0
        self.source: str | None = None
        self.name: str | None = None
        self.tilewidth: int = 0
        self.tileheight: int = 0
        self.spacing: int = 0
        self.margin: int = 0
        self.tilecount: int = 0
        self.columns: int = 0

        # image properties
        self.trans: str | None = None
        self.width: int = 0
        self.height: int = 0

        self.parse_xml(node)

    def _resolve_path(self, path: str, relative_to_source: bool) -> str:
        """
        Resolve a path relative to either the TMX or TSX file, but keep it relative.
        """
        base_path = self.tileset_source if relative_to_source else self.parent.filename
        base = os.path.dirname(base_path or "")
        resolved = os.path.join(base, path)
        logger.debug(f"Resolved path: {resolved}")
        return resolved

    def _parse_tile_properties(self, node: ElementTree.Element) -> dict[str, Any]:
        """
        Parses a single tile's attributes and custom properties.
        """
        props = {k: types[k](v) for k, v in node.items()}
        props.update(parse_properties(node))
        logger.debug(f"Parsed tile properties: {props}")
        return props

    def _parse_animation_frames(
        self, anim_node: ElementTree.Element
    ) -> list[AnimationFrame]:
        """
        Parses animation frames from a tile's animation node.
        """
        frames = []
        for frame in anim_node.findall("frame"):
            duration = int(frame.get("duration") or 0)
            gid = self.parent.register_gid(
                int(frame.get("tileid") or 0) + self.firstgid
            )
            frames.append(AnimationFrame(gid, duration))
            logger.debug(
                f"Parsed animation frame: tileid={frame.get('tileid')}, duration={duration}, gid={gid}"
            )
        return frames

    def _handle_external_source(self, node: ElementTree.Element) -> ElementTree.Element:
        """Handles parsing an external tileset source."""
        source = node.get("source", None)
        if not source:
            return node

        if source[-4:].lower() != ".tsx":
            msg = f"Found external tileset, but cannot handle type: {source}"
            logger.error(msg)
            raise ValueError(msg)

        self.tileset_source = source
        self.firstgid = int(node.get("firstgid") or 0)
        logger.debug(f"External tileset detected: {source}, firstgid={self.firstgid}")

        resolved_path = self._resolve_path(source, relative_to_source=False)

        try:
            new_node = ElementTree.parse(resolved_path).getroot()
            logger.debug(f"Successfully loaded external tileset from {resolved_path}")
            return new_node
        except FileNotFoundError as e:
            msg = f"Cannot find tileset file {source} from {self.parent.filename}, should be at {resolved_path}"
            logger.error(msg)
            raise FileNotFoundError(msg) from e
        except ParseError as e:
            msg = f"Error loading external tileset: {resolved_path}"
            logger.error(msg)
            raise ParseError(msg) from e

    def _parse_all_tiles(self, node: ElementTree.Element, is_external: bool) -> None:
        """Parses all individual tiles within the tileset node."""
        for child in node.iter("tile"):
            raw_id = child.get("id")
            if raw_id is None:
                raise TypeError("Tile is missing required 'id' attribute")

            tiled_gid = int(raw_id)
            props = self._parse_tile_properties(child)
            logger.debug(f"Parsing tile ID: {tiled_gid}")

            image_node = child.find("image")
            if image_node is not None:
                tile_source = image_node.get("source")
                if tile_source and is_external:
                    tile_source = self._resolve_path(
                        tile_source, relative_to_source=True
                    )
                props["source"] = tile_source
                props["trans"] = image_node.get("trans", None)
                props["width"] = int(image_node.get("width") or 0)
                props["height"] = int(image_node.get("height") or 0)
                logger.debug(
                    f"Tile image parsed: source={tile_source}, size={props['width']}x{props['height']}"
                )
            else:
                props["width"] = self.tilewidth
                props["height"] = self.tileheight

            anim = child.find("animation")
            props["frames"] = (
                self._parse_animation_frames(anim) if anim is not None else []
            )
            colliders = []

            logger.debug(f"Object group parsed for tile ID {tiled_gid}")
            for objgrp_node in child.findall("objectgroup"):
                for obj in objgrp_node.findall("object"):
                    shape = "rectangle"
                    collider = {
                        "x": float(obj.get("x", 0)),
                        "y": float(obj.get("y", 0)),
                        "rotation": float(obj.get("rotation", 0) or 0),
                        "width": 0.0,
                        "height": 0.0,
                        "type": shape,
                        "points": [],
                    }

                    polygon_node = obj.find("polygon")
                    if polygon_node is not None:
                        points_str = polygon_node.get("points", "")
                        collider["points"] = [
                            tuple(map(float, p.strip().split(",")))
                            for p in points_str.strip().split(" ")
                        ]

                    elif obj.find("ellipse") is not None:
                        shape = "ellipse"
                        collider["type"] = shape
                        collider["width"] = float(obj.get("width", 0))
                        collider["height"] = float(obj.get("height", 0))

                    elif obj.find("point") is not None:
                        shape = "point"
                        collider["type"] = shape

                    else:
                        collider["width"] = float(obj.get("width", 0))
                        collider["height"] = float(obj.get("height", 0))

                    colliders.append(collider)

            props["colliders"] = colliders

            for gid, _flags in self.parent.map_gid2(tiled_gid + self.firstgid):
                self.parent.set_tile_properties(gid, props)

    def _parse_tileset_image_and_offset(self, node: ElementTree.Element) -> None:
        """Parses the main tileset image and offset nodes."""
        tile_offset_node = node.find("tileoffset")
        if tile_offset_node is not None:
            self.offset = (
                int(tile_offset_node.get("x") or 0),
                int(tile_offset_node.get("y") or 0),
            )
            logger.debug(f"Parsed tileoffset: {self.offset}")

        image_node = node.find("image")
        if image_node is not None:
            self.source = image_node.get("source")
            if self.source and self.tileset_source:
                self.source = self._resolve_path(self.source, relative_to_source=True)
            self.trans = image_node.get("trans", None)
            self.width = int(image_node.get("width") or 0)
            self.height = int(image_node.get("height") or 0)
            logger.debug(
                f"Tileset image node parsed: source={self.source}, size={self.width}x{self.height}, trans={self.trans}"
            )

    def parse_xml(self, node: ElementTree.Element) -> Self:
        """
        Parse a TiledTileset layer from ElementTree xml node.

        Returns:
            TiledTileset: The parsed TiledTileset layer.
        """
        logger.debug("Starting XML parsing for tileset")

        original_source = node.get("source")

        # Handle external tileset source
        node = self._handle_external_source(node)

        # Set main tileset properties
        self._set_properties(node)
        logger.debug(
            f"Tileset properties set: name={self.name}, tilecount={self.tilecount}, columns={self.columns}"
        )

        # Parse all individual tiles
        self._parse_all_tiles(node, bool(original_source))

        # Parse main tileset image and offset
        self._parse_tileset_image_and_offset(node)

        logger.debug("Finished parsing tileset XML")
        return self
