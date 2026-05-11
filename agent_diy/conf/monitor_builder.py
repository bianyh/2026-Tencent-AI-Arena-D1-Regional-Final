#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()
    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="round(avg(reward{}), 0.01)")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="round(avg(total_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="round(avg(value_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="round(avg(policy_loss{}), 0.01)")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="round(avg(entropy_loss{}), 0.01)")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
