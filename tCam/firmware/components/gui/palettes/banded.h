/*
 * Banded colormap for 8-bit LEP indexed data
 *
 * Copyright 2021 Dan Julio
 *
 * This file is part of firecam.
 *
 * firecam is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * firecam is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with firecam.  If not, see <https://www.gnu.org/licenses/>.
 *
 */
#ifndef PALETTE_BANDED_H
#define PALETTE_BANDED_H

#include "palettes.h"

palette_map_t banded_palette_map = {
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{0,0,0},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{96,0,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,80},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{48,48,112},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{80,80,128},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{96,96,176},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{112,112,192},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{128,128,224},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,96,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{48,144,48},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{80,192,80},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{64,224,64},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{224,224,80},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,208,96},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,176,64},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{208,144,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{192,96,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{176,48,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,0,0},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
	{255,255,255},
};

#endif