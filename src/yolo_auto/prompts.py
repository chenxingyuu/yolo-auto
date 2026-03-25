from __future__ import annotations

from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field


def register_prompts(mcp: FastMCP) -> None:
    """Register MCP Prompts for one-shot CEO-friendly workflows."""

    @mcp.prompt(name="quick-train")
    def quick_train_prompt(
        dataset: Annotated[
            str,
            Field(description="数据集 YAML 路径，如 /workspace/datasets/coco.yaml"),
        ],
        model: Annotated[
            str,
            Field(description="模型权重路径或名称，如 yolov8n.pt 或 /workspace/models/yolo26n.pt"),
        ] = "yolov8n.pt",
        epochs: Annotated[
            str,
            Field(description="训练轮数"),
        ] = "100",
    ) -> str:
        """一句话启动训练：环境检查 → 启动 → 首次状态确认。"""
        return (
            f"请帮我完成以下 YOLO 训练流程（按顺序执行，每步失败立即告知原因与修复建议）：\n"
            f"\n"
            f"0. 路径约定（远程容器内）：\n"
            f"   - 数据集 YAML 通常在 /workspace/datasets（例：/workspace/datasets/coco.yaml）\n"
            f"   - 基础模型权重通常在 /workspace/models（例：/workspace/models/yolo26n.pt）\n"
            f"\n"
            f"1. 调用 yolo_setup_env（dataConfigPath=\"{dataset}\"）确认远程环境就绪\n"
            f"\n"
            f"2. 【强制确认】在调用 yolo_start_training 之前：\n"
            f"   - 先把将要启动的配置完整列出来：\n"
            f"     * model = \"{model}\"\n"
            f"     * dataConfigPath = \"{dataset}\"\n"
            f"     * epochs = {epochs}\n"
            f"     * imgSize = 640, batch = 16, learningRate = 0.01\n"
            f"   - 再给出等价的 Ultralytics 命令示例（便于人工核对，不要求完全逐字一致）：\n"
            f"     yolo detect train model=\"{model}\" data=\"{dataset}\" epochs={epochs} "
            f"imgsz=640 batch=16 lr0=0.01\n"
            f"   - 明确要求用户回复“确认/开始”后再继续\n"
            f"   - 在收到用户确认前，不要调用 yolo_start_training\n"
            f"\n"
            f"3. 收到用户确认后，调用 yolo_start_training 启动训练（参数同上）\n"
            f"4. 启动成功后等待约 30 秒，调用 yolo_get_status 确认训练已开始运行\n"
            f"5. 用一段简洁摘要回复：任务 ID、MLflow runId、关键参数、飞书是否已通知\n"
        )

    @mcp.prompt(name="dashboard")
    def dashboard_prompt() -> str:
        """全局状态看板：一眼掌握所有训练任务进展。"""
        return (
            "请帮我生成当前训练状态看板：\n"
            "\n"
            "1. 调用 yolo_list_jobs 获取最近所有任务\n"
            "2. 对状态为 running 的任务逐个调用 yolo_get_job（refresh=true）刷新指标\n"
            "3. 用表格汇总：任务 ID | 状态 | 模型 | 当前 epoch/总 epoch | 主指标 | 启动时间\n"
            "4. 运行中的任务给出预估剩余时间\n"
            "5. 失败的任务简述原因\n"
            "6. 最后用一句话总结全局情况（如「2 个运行中、1 个已完成、最佳 mAP 72.3%」）\n"
        )

    @mcp.prompt(name="compare-experiments")
    def compare_experiments_prompt(
        job_ids: Annotated[
            str,
            Field(description="要对比的任务 ID，用逗号分隔，如 job-001,job-002,job-003"),
        ],
    ) -> str:
        """对比多个实验并给出最佳推荐。job_ids 逗号分隔。"""
        ids = [jid.strip() for jid in job_ids.split(",")]
        job_list = "\n".join(f"   - {jid}" for jid in ids)
        return (
            f"请对比以下训练实验并给出推荐：\n"
            f"\n"
            f"1. 逐个调用 yolo_get_job（refresh=true）获取详情：\n"
            f"{job_list}\n"
            f"2. 输出对比表格：模型 | epochs | batch | lr | 主指标(mAP) | 训练时长 | 状态\n"
            f"3. 分析：\n"
            f"   - 哪个指标最好？领先幅度？\n"
            f"   - 参数差异对结果的关键影响\n"
            f"   - 是否有过拟合/欠拟合迹象\n"
            f"4. 给出明确结论：推荐哪个方案，以及下一步行动建议（继续调参/增加 epoch/换模型）\n"
        )

    @mcp.prompt(name="smart-tune")
    def smart_tune_prompt(
        dataset: Annotated[
            str,
            Field(description="数据集 YAML 路径，如 /workspace/datasets/coco.yaml"),
        ],
        model: Annotated[
            str,
            Field(description="基础模型权重路径或名称"),
        ] = "yolov8n.pt",
        goal: Annotated[
            str,
            Field(description="优化目标：如「精度优先」「速度优先」「精度与速度兼顾」"),
        ] = "精度与速度兼顾",
    ) -> str:
        """根据业务目标智能推荐调参策略并一键执行。"""
        return (
            f"请帮我制定并执行智能调参方案：\n"
            f"\n"
        f"路径约定（远程容器内）：\n"
        f"- 数据集 YAML 通常在 /workspace/datasets（例：/workspace/datasets/coco.yaml）\n"
        f"- 基础模型权重通常在 /workspace/models（例：/workspace/models/yolo26n.pt）\n"
        f"\n"
            f"目标：{goal}\n"
            f"模型：{model}\n"
            f"数据集：{dataset}\n"
            f"\n"
            f"1. 调用 yolo_setup_env 确认环境\n"
            f"2. 调用 yolo_list_jobs 查看历史实验，避免重复参数\n"
            f"3. 根据目标「{goal}」设计搜索空间：\n"
            f"   - 追求精度 → imgSize [640,1280], batch [8,16], lr [0.001,0.01]\n"
            f"   - 追求速度 → imgSize [320,640], batch [32,64], lr [0.01,0.02]\n"
            f"   - 兼顾平衡 → imgSize [640], batch [16,32], lr [0.005,0.01,0.02]\n"
        f"\n"
        f"4. 【强制确认】在调用 yolo_auto_tune 之前：\n"
        f"   - 先总结你将要执行的调参计划（必须具体可核对）：\n"
        f"     * baseJobId（你将生成的 trial 前缀）\n"
        f"     * model = \"{model}\"\n"
        f"     * dataConfigPath = \"{dataset}\"\n"
        f"     * epochs = 30（快速筛选）\n"
        f"     * maxTrials <= 6\n"
        f"     * 你设计的搜索空间（imgSize/batch/lr 等）\n"
        f"   - 再给出“将要调用 yolo_auto_tune 的参数 JSON 概览”（字段名需与 tool 入参一致）\n"
        f"   - 明确要求用户回复“确认/开始”后再继续\n"
        f"   - 在收到用户确认前，不要调用 yolo_auto_tune\n"
        f"\n"
        f"5. 收到用户确认后，调用 yolo_auto_tune（epochs=30、maxTrials<=6）执行调参\n"
        f"6. 输出报告：最佳参数、各 trial 对比、是否建议用最佳参数跑完整 epoch\n"
        )

    @mcp.prompt(name="diagnose")
    def diagnose_prompt(
        job_id: Annotated[
            str,
            Field(description="需要诊断的训练任务 ID"),
        ],
    ) -> str:
        """诊断训练任务异常：检查 → 定位 → 给出修复方案。"""
        return (
            f"任务 {job_id} 可能出了问题，请帮我诊断：\n"
            f"\n"
            f"1. 调用 yolo_get_job（jobId=\"{job_id}\", refresh=true）获取最新状态与指标\n"
            f"2. 根据状态判断：\n"
            f"   - running 但长时间无新 epoch → 是否卡住（GPU 挂起/OOM 后静默失败）\n"
            f"   - failed → 分析错误（OOM、数据路径、SSH 断连、磁盘满）\n"
            f"   - completed 但指标差 → 欠拟合（epoch 不够/lr 太小）或过拟合（train↑ val↓）\n"
            f"3. 给出诊断结论和修复方案\n"
            f"4. 如果需要重跑，直接给出调整后的推荐参数\n"
        )

    @mcp.prompt(name="report")
    def report_prompt(
        period: Annotated[
            str,
            Field(description="报告周期，如「今天」「本周」「最近 3 天」"),
        ] = "今天",
    ) -> str:
        """生成可直接转发给团队/上级的训练进展报告。"""
        return (
            f"请帮我生成「{period}」的训练进展报告（适合发飞书群/汇报）：\n"
            f"\n"
            f"1. 调用 yolo_list_jobs 获取相关任务\n"
            f"2. 对运行中/已完成的任务调用 yolo_get_job（refresh=true）刷新数据\n"
            f"3. 按以下结构输出报告：\n"
            f"\n"
            f"   【训练进展摘要】\n"
            f"   - 本轮共运行 X 个实验\n"
            f"   - 当前最佳：[模型] mAP=XX.X%（任务 ID）\n"
            f"   - 相比上次提升/下降 X 个百分点\n"
            f"\n"
            f"   【实验明细】\n"
            f"   （表格：任务 ID | 模型 | 关键参数 | 主指标 | 状态）\n"
            f"\n"
            f"   【下一步计划】\n"
            f"   - 基于当前结果的优化方向\n"
            f"   - 需要的资源或配置调整\n"
            f"\n"
            f"报告要简洁专业、数据准确、结论清晰。\n"
        )


