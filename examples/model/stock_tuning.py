"""
股票咨询模型微调示例
"""
import asyncio
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer, TrainingArguments, Trainer

from base.model.tuning import (DataProcessor, HFModelTuner, TuningCallback,
                             TuningConfig, TuningMethod, TuningMetrics)


class StockDataProcessor(DataProcessor):
    """股票数据处理器"""
    
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    async def process(
        self,
        data: List[Dict[str, str]],
        config: TuningConfig
    ) -> Dict[str, Any]:
        """处理股票咨询数据"""
        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]
        
        # 标记化
        inputs = await self.tokenize(questions, config)
        labels = await self.tokenize(answers, config)
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"]
        }
    
    async def tokenize(
        self,
        texts: Union[str, List[str]],
        config: TuningConfig
    ) -> Dict[str, Any]:
        """文本标记化"""
        if isinstance(texts, str):
            texts = [texts]
            
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt"
        )
    
    async def create_dataset(
        self,
        data: Any,
        config: TuningConfig
    ) -> Any:
        """创建数据集"""
        processed = await self.process(data, config)
        
        from torch.utils.data import Dataset
        
        class StockDataset(Dataset):
            def __init__(self, data):
                self.data = data
                
            def __len__(self):
                return len(self.data["input_ids"])
                
            def __getitem__(self, idx):
                return {
                    key: val[idx]
                    for key, val in self.data.items()
                }
                
        return StockDataset(processed)


class MetricsCallback(TuningCallback):
    """指标记录回调"""
    
    def __init__(self):
        self.metrics_history: List[TuningMetrics] = []
    
    async def on_epoch_begin(
        self,
        epoch: int,
        logs: Dict[str, Any]
    ) -> None:
        """记录epoch开始"""
        print(f"开始第 {epoch} 个epoch")
    
    async def on_epoch_end(
        self,
        epoch: int,
        metrics: TuningMetrics
    ) -> None:
        """记录epoch结束"""
        self.metrics_history.append(metrics)
        print(f"第 {epoch} 个epoch结束，loss: {metrics.loss:.4f}")
    
    async def on_batch_begin(
        self,
        batch: int,
        logs: Dict[str, Any]
    ) -> None:
        """记录batch开始"""
        pass
    
    async def on_batch_end(
        self,
        batch: int,
        metrics: TuningMetrics
    ) -> None:
        """记录batch结束"""
        if batch % 100 == 0:
            print(f"完成第 {batch} 个batch，loss: {metrics.loss:.4f}")


class StockModelTuner(HFModelTuner):
    """股票模型微调器"""
    
    async def prepare_model(self) -> None:
        """准备模型"""
        from transformers import AutoModelForCausalLM
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name
        )
        
        if self.config.method == TuningMethod.LORA:
            from peft import get_peft_model, LoraConfig
            
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=32,
                lora_dropout=0.1
            )
            
            self.model = get_peft_model(self.model, peft_config)
    
    async def train(
        self,
        train_data: Any,
        eval_data: Optional[Any] = None
    ) -> List[TuningMetrics]:
        """训练模型"""
        # 创建训练参数
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data
        )
        
        # 训练模型
        metrics_history = []
        
        for epoch in range(self.config.num_epochs):
            # 通知epoch开始
            for callback in self.callbacks:
                await callback.on_epoch_begin(epoch, {})
            
            # 训练一个epoch
            train_output = trainer.train()
            
            # 创建指标
            metrics = TuningMetrics(
                loss=train_output.training_loss,
                accuracy=0.0,  # 需要实现准确率计算
                epoch=epoch,
                step=train_output.global_step,
                learning_rate=trainer.optimizer.param_groups[0]["lr"],
                batch_size=self.config.batch_size
            )
            
            metrics_history.append(metrics)
            
            # 通知epoch结束
            for callback in self.callbacks:
                await callback.on_epoch_end(epoch, metrics)
        
        return metrics_history
    
    async def evaluate(
        self,
        eval_data: Any
    ) -> TuningMetrics:
        """评估模型"""
        # 创建评估器
        trainer = Trainer(
            model=self.model,
            eval_dataset=eval_data
        )
        
        # 评估模型
        eval_output = trainer.evaluate()
        
        return TuningMetrics(
            loss=eval_output["eval_loss"],
            accuracy=0.0,  # 需要实现准确率计算
            epoch=-1,
            step=-1,
            learning_rate=0.0,
            batch_size=self.config.batch_size
        )
    
    async def save(
        self,
        path: str
    ) -> None:
        """保存模型"""
        self.model.save_pretrained(path)
    
    async def load(
        self,
        path: str
    ) -> None:
        """加载模型"""
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(path)


async def main():
    """主函数"""
    # 准备示例数据
    train_data = [
        {
            "question": "请查询贵州茅台的最新股价",
            "answer": "当前贵州茅台(600519)的价格是2000元，较前一交易日上涨1.5%。"
        },
        {
            "question": "分析一下贵州茅台最近一周的走势",
            "answer": "过去一周贵州茅台呈现震荡上行趋势，成交量有所放大，主力资金持续流入。"
        }
        # 添加更多训练数据...
    ]
    
    # 创建配置
    config = TuningConfig(
        method=TuningMethod.LORA,
        model_name="baichuan-inc/Baichuan2-7B-Chat",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )
    
    # 创建处理器
    processor = StockDataProcessor(config.model_name)
    
    # 创建回调
    callback = MetricsCallback()
    
    # 创建微调器
    tuner = StockModelTuner(
        config=config,
        processor=processor,
        callbacks=[callback]
    )
    
    # 准备数据集
    dataset = await processor.create_dataset(train_data, config)
    
    # 准备模型
    await tuner.prepare_model()
    
    # 训练模型
    metrics = await tuner.train(dataset)
    
    # 保存模型
    await tuner.save("./stock_model")
    
    print("训练完成！")
    print(f"最终loss: {metrics[-1].loss:.4f}")


if __name__ == "__main__":
    asyncio.run(main()) 