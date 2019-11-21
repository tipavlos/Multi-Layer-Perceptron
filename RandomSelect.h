#pragma once

static unsigned int g_seed = 1234567;
static inline int fastrand()
{
	g_seed = (214013 * g_seed + 2531011);
	return (g_seed >> 16) & 0x7FFF;
}