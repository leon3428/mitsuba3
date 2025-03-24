#include <mitsuba/core/class.h>
#include <mitsuba/core/config.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/ltm.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/xml.h>
#include <nanothread/nanothread.h>
#include <pugixml.hpp>

NAMESPACE_BEGIN(mitsuba)
NAMESPACE_BEGIN(ltm)

int add(int a, int b) { return a + b; }

NAMESPACE_END(ltm)
NAMESPACE_END(mitsuba)
