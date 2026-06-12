#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ===============================================================================
#
# Copyright (c) 2026 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved
#
#
# File: /d/Academia/Courses/ML26Q2/courses/04_classcodes/extended/02_py_unittest/weatherforcast.py
# Author: Hai Liang Wang
# Date: 2026-06-01:10:17:16
#
# ===============================================================================

"""
介绍 Python 的单元测试
"""
__copyright__ = "Copyright (c) 2026 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2026-06-01:10:17:16"

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))


def weather_forecast(wind_speed):

    if wind_speed > 1 and wind_speed <= 10:
        return "晴朗"
    elif wind_speed > 10:
        return "要下雨"

