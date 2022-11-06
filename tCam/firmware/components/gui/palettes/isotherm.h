/*
 * "IsoTherm" colormap for 8-bit LEP indexed data
 *
 * Copyright 2022 Dan Julio
 *
 * This file is part of tCam.
 *
 * tCam is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * tCam is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with tCam.  If not, see <https://www.gnu.org/licenses/>.
 *
 */
#ifndef PALETTE_ISOTHERM_H
#define PALETTE_ISOTHERM_H

#include "palettes.h"

palette_map_t  isotherm_palette_map = {
	{0, 15, 255},
	{3, 18, 247},
	{6, 21, 239},
	{9, 23, 231},
	{12, 25, 223},
	{14, 27, 215},
	{17, 29, 207},
	{19, 30, 199},
	{21, 31, 191},
	{22, 32, 183},
	{24, 33, 175},
	{25, 34, 167},
	{26, 34, 159},
	{27, 35, 151},
	{27, 35, 143},
	{28, 34, 135},
	{28, 34, 127},
	{28, 33, 119},
	{27, 33, 111},
	{27, 32, 103},
	{26, 30, 95},
	{25, 29, 87},
	{24, 27, 79},
	{22, 25, 71},
	{21, 23, 63},
	{19, 21, 55},
	{17, 19, 47},
	{14, 16, 39},
	{12, 13, 31},
	{9, 10, 23},
	{6, 7, 15},
	{3, 3, 7},
	{0, 0, 0},
	{1, 1, 1},
	{2, 2, 2},
	{3, 3, 3},
	{4, 4, 4},
	{5, 5, 5},
	{6, 6, 6},
	{7, 7, 7},
	{9, 9, 9},
	{10, 10, 10},
	{11, 11, 11},
	{12, 12, 12},
	{13, 13, 13},
	{14, 14, 14},
	{15, 15, 15},
	{17, 17, 17},
	{18, 18, 18},
	{19, 19, 19},
	{20, 20, 20},
	{21, 21, 21},
	{22, 22, 22},
	{23, 23, 23},
	{25, 25, 25},
	{26, 26, 26},
	{27, 27, 27},
	{28, 28, 28},
	{29, 29, 29},
	{30, 30, 30},
	{31, 31, 31},
	{33, 33, 33},
	{34, 34, 34},
	{35, 35, 35},
	{36, 36, 36},
	{37, 37, 37},
	{38, 38, 38},
	{39, 39, 39},
	{40, 40, 40},
	{42, 42, 42},
	{43, 43, 43},
	{44, 44, 44},
	{45, 45, 45},
	{46, 46, 46},
	{47, 47, 47},
	{48, 48, 48},
	{50, 50, 50},
	{51, 51, 51},
	{52, 52, 52},
	{53, 53, 53},
	{54, 54, 54},
	{55, 55, 55},
	{56, 56, 56},
	{58, 58, 58},
	{59, 59, 59},
	{60, 60, 60},
	{61, 61, 61},
	{62, 62, 62},
	{63, 63, 63},
	{64, 64, 64},
	{66, 66, 66},
	{67, 67, 67},
	{68, 68, 68},
	{69, 69, 69},
	{70, 70, 70},
	{71, 71, 71},
	{72, 72, 72},
	{73, 73, 73},
	{75, 75, 75},
	{76, 76, 76},
	{77, 77, 77},
	{78, 78, 78},
	{79, 79, 79},
	{80, 80, 80},
	{81, 81, 81},
	{83, 83, 83},
	{84, 84, 84},
	{85, 85, 85},
	{86, 86, 86},
	{87, 87, 87},
	{88, 88, 88},
	{89, 89, 89},
	{91, 91, 91},
	{92, 92, 92},
	{93, 93, 93},
	{94, 94, 94},
	{95, 95, 95},
	{96, 96, 96},
	{97, 97, 97},
	{99, 99, 99},
	{100, 100, 100},
	{101, 101, 101},
	{102, 102, 102},
	{103, 103, 103},
	{104, 104, 104},
	{105, 105, 105},
	{107, 107, 107},
	{108, 108, 108},
	{109, 109, 109},
	{110, 110, 110},
	{111, 111, 111},
	{112, 112, 112},
	{113, 113, 113},
	{114, 114, 114},
	{116, 116, 116},
	{117, 117, 117},
	{118, 118, 118},
	{119, 119, 119},
	{120, 120, 120},
	{121, 121, 121},
	{122, 122, 122},
	{124, 124, 124},
	{125, 125, 125},
	{126, 126, 126},
	{127, 127, 127},
	{128, 128, 128},
	{129, 129, 129},
	{130, 130, 130},
	{132, 132, 132},
	{133, 133, 133},
	{134, 134, 134},
	{135, 135, 135},
	{136, 136, 136},
	{137, 137, 137},
	{138, 138, 138},
	{140, 140, 140},
	{141, 141, 141},
	{142, 142, 142},
	{143, 143, 143},
	{144, 144, 144},
	{145, 145, 145},
	{146, 146, 146},
	{147, 147, 147},
	{149, 149, 149},
	{150, 150, 150},
	{151, 151, 151},
	{152, 152, 152},
	{153, 153, 153},
	{154, 154, 154},
	{155, 155, 155},
	{157, 157, 157},
	{158, 158, 158},
	{159, 159, 159},
	{160, 160, 160},
	{161, 161, 161},
	{162, 162, 162},
	{163, 163, 163},
	{165, 165, 165},
	{166, 166, 166},
	{167, 167, 167},
	{168, 168, 168},
	{169, 169, 169},
	{170, 170, 170},
	{171, 171, 171},
	{173, 173, 173},
	{174, 174, 174},
	{175, 175, 175},
	{176, 176, 176},
	{177, 177, 177},
	{178, 178, 178},
	{179, 179, 179},
	{181, 181, 181},
	{182, 182, 182},
	{183, 183, 183},
	{184, 184, 184},
	{185, 185, 185},
	{186, 186, 186},
	{187, 187, 187},
	{188, 188, 188},
	{190, 190, 190},
	{191, 191, 191},
	{192, 192, 192},
	{193, 193, 193},
	{194, 194, 194},
	{195, 195, 195},
	{196, 196, 196},
	{198, 198, 198},
	{199, 199, 199},
	{200, 200, 200},
	{201, 201, 201},
	{202, 202, 202},
	{203, 203, 203},
	{204, 204, 204},
	{206, 206, 206},
	{207, 207, 207},
	{208, 208, 208},
	{209, 209, 209},
	{210, 210, 210},
	{211, 211, 211},
	{212, 212, 212},
	{214, 214, 214},
	{215, 215, 215},
	{216, 216, 216},
	{217, 217, 217},
	{218, 218, 218},
	{218, 211, 211},
	{218, 204, 204},
	{218, 198, 198},
	{218, 191, 191},
	{218, 184, 184},
	{218, 177, 177},
	{218, 170, 170},
	{218, 163, 163},
	{218, 157, 157},
	{218, 150, 150},
	{218, 143, 143},
	{218, 136, 136},
	{218, 129, 129},
	{218, 122, 122},
	{218, 116, 116},
	{218, 109, 109},
	{218, 102, 102},
	{218, 95, 95},
	{218, 88, 88},
	{218, 81, 81},
	{218, 75, 75},
	{218, 68, 68},
	{218, 61, 61},
	{218, 54, 54},
	{218, 47, 47},
	{218, 40, 40},
	{218, 34, 34},
	{218, 27, 27},
	{218, 20, 20},
	{218, 13, 13},
	{218, 6, 6}
};

#endif
