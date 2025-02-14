"""
股票交易强化学习示例
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from base.model.rl import (Environment, RLAgent, RLAlgorithm, RLCallback,
                          RLConfig, RLMetrics, RewardFunction, RewardType,
                          SB3Agent)


class StockTradingEnv(Environment):
    """股票交易环境"""
    
    def __init__(
        self,
        data: List[Dict[str, float]],
        initial_balance: float = 10000.0
    ):
        self.data = data
        self.initial_balance = initial_balance
        self._current_step = 0
        self._balance = initial_balance
        self._position = 0  # 持仓数量
        
        # 定义动作空间：-1(卖出), 0(持有), 1(买入)
        self._action_space = np.array([-1, 0, 1])
        
        # 定义观察空间：[当前价格, 持仓数量, 账户余额]
        self._observation_space = np.array([np.inf, np.inf, np.inf])
    
    @property
    def action_space(self) -> np.ndarray:
        return self._action_space
    
    @property
    def observation_space(self) -> np.ndarray:
        return self._observation_space
    
    async def reset(self) -> np.ndarray:
        """重置环境"""
        self._current_step = 0
        self._balance = self.initial_balance
        self._position = 0
        return self._get_observation()
    
    async def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行交易动作"""
        current_price = self.data[self._current_step]["close"]
        
        # 执行交易
        if action == 1:  # 买入
            shares = self._balance // current_price
            if shares > 0:
                self._position += shares
                self._balance -= shares * current_price
        elif action == -1:  # 卖出
            if self._position > 0:
                self._balance += self._position * current_price
                self._position = 0
                
        # 更新状态
        self._current_step += 1
        done = self._current_step >= len(self.data) - 1
        
        # 计算收益
        next_price = self.data[self._current_step]["close"]
        portfolio_value = self._balance + self._position * next_price
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        return self._get_observation(), reward, done, {}
    
    async def render(self) -> Dict[str, float]:
        """返回当前状态"""
        current_price = self.data[self._current_step]["close"]
        portfolio_value = self._balance + self._position * current_price
        return {
            "step": self._current_step,
            "price": current_price,
            "balance": self._balance,
            "position": self._position,
            "portfolio_value": portfolio_value
        }
    
    async def close(self) -> None:
        """关闭环境"""
        pass
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        current_price = self.data[self._current_step]["close"]
        return np.array([
            current_price,
            self._position,
            self._balance
        ])


class PortfolioRewardFunction(RewardFunction):
    """投资组合奖励函数"""
    
    async def calculate(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """计算基础奖励"""
        current_value = state[1] * state[0] + state[2]  # position * price + balance
        next_value = next_state[1] * next_state[0] + next_state[2]
        return (next_value - current_value) / current_value
    
    async def shape(
        self,
        reward: float,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """整形奖励"""
        # 添加交易成本惩罚
        if action != 0:  # 如果发生交易
            reward -= 0.001  # 0.1%交易成本
        return reward


class TradingMetricsCallback(RLCallback):
    """交易指标回调"""
    
    def __init__(self):
        self.episode_rewards = []
        self.portfolio_values = []
    
    async def on_episode_begin(
        self,
        episode: int,
        logs: Dict[str, Any]
    ) -> None:
        """记录episode开始"""
        print(f"开始第 {episode} 个episode")
    
    async def on_episode_end(
        self,
        episode: int,
        metrics: RLMetrics
    ) -> None:
        """记录episode结束"""
        self.episode_rewards.append(metrics.reward_mean)
        print(f"第 {episode} 个episode结束，平均奖励: {metrics.reward_mean:.4f}")
    
    async def on_step_begin(
        self,
        step: int,
        logs: Dict[str, Any]
    ) -> None:
        """记录step开始"""
        pass
    
    async def on_step_end(
        self,
        step: int,
        metrics: RLMetrics
    ) -> None:
        """记录step结束"""
        if step % 1000 == 0:
            print(f"完成第 {step} 步，当前奖励: {metrics.reward_mean:.4f}")


class StockTradingAgent(SB3Agent):
    """股票交易智能体"""
    
    async def build_model(self) -> None:
        """构建PPO模型"""
        self.model = PPO(
            "MlpPolicy",
            DummyVecEnv([lambda: self.env]),
            learning_rate=self.config.learning_rate,
            n_steps=self.config.update_interval,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            verbose=1
        )
    
    async def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """选择交易动作"""
        action, _ = self.model.predict(state, deterministic=not training)
        return int(action)
    
    async def train(self) -> List[RLMetrics]:
        """训练模型"""
        metrics_history = []
        
        for episode in range(self.config.num_episodes):
            # 通知episode开始
            for callback in self.callbacks:
                await callback.on_episode_begin(episode, {})
            
            state = await self.env.reset()
            episode_rewards = []
            
            for step in range(self.config.max_steps_per_episode):
                # 通知step开始
                for callback in self.callbacks:
                    await callback.on_step_begin(step, {})
                
                # 选择动作
                action = await self.select_action(state)
                
                # 执行动作
                next_state, reward, done, info = await self.env.step(action)
                
                # 如果有奖励函数，使用它来计算和整形奖励
                if self.reward_func:
                    reward = await self.reward_func.calculate(
                        state, action, next_state, info
                    )
                    reward = await self.reward_func.shape(
                        reward, state, action, next_state, info
                    )
                
                episode_rewards.append(reward)
                
                # 更新模型
                self.model.learn(total_timesteps=1)
                
                # 创建step指标
                step_metrics = RLMetrics(
                    episode=episode,
                    total_steps=step,
                    reward_mean=np.mean(episode_rewards),
                    reward_std=np.std(episode_rewards),
                    value_loss=0.0,  # 需要从模型获取
                    policy_loss=0.0,  # 需要从模型获取
                    entropy=0.0,      # 需要从模型获取
                    learning_rate=self.config.learning_rate
                )
                
                # 通知step结束
                for callback in self.callbacks:
                    await callback.on_step_end(step, step_metrics)
                
                if done:
                    break
                    
                state = next_state
            
            # 创建episode指标
            episode_metrics = RLMetrics(
                episode=episode,
                total_steps=step,
                reward_mean=np.mean(episode_rewards),
                reward_std=np.std(episode_rewards),
                value_loss=0.0,
                policy_loss=0.0,
                entropy=0.0,
                learning_rate=self.config.learning_rate
            )
            
            metrics_history.append(episode_metrics)
            
            # 通知episode结束
            for callback in self.callbacks:
                await callback.on_episode_end(episode, episode_metrics)
        
        return metrics_history
    
    async def evaluate(
        self,
        num_episodes: int = 10
    ) -> RLMetrics:
        """评估模型"""
        all_rewards = []
        
        for episode in range(num_episodes):
            state = await self.env.reset()
            episode_rewards = []
            
            while True:
                action = await self.select_action(state, training=False)
                next_state, reward, done, _ = await self.env.step(action)
                episode_rewards.append(reward)
                
                if done:
                    break
                    
                state = next_state
            
            all_rewards.extend(episode_rewards)
        
        return RLMetrics(
            episode=-1,
            total_steps=len(all_rewards),
            reward_mean=np.mean(all_rewards),
            reward_std=np.std(all_rewards),
            value_loss=0.0,
            policy_loss=0.0,
            entropy=0.0,
            learning_rate=0.0
        )
    
    async def save(
        self,
        path: str
    ) -> None:
        """保存模型"""
        self.model.save(path)
    
    async def load(
        self,
        path: str
    ) -> None:
        """加载模型"""
        self.model = PPO.load(path)


async def main():
    """主函数"""
    # 创建模拟数据
    data = []
    price = 100.0
    for i in range(1000):
        price *= (1 + np.random.normal(0, 0.02))  # 2%的每日波动率
        data.append({
            "date": datetime.now() + timedelta(days=i),
            "open": price * (1 + np.random.normal(0, 0.005)),
            "high": price * (1 + np.random.normal(0, 0.005)),
            "low": price * (1 + np.random.normal(0, 0.005)),
            "close": price,
            "volume": np.random.randint(1000, 10000)
        })
    
    # 创建环境
    env = StockTradingEnv(data)
    
    # 创建奖励函数
    reward_func = PortfolioRewardFunction()
    
    # 创建回调
    callback = TradingMetricsCallback()
    
    # 创建配置
    config = RLConfig(
        algorithm=RLAlgorithm.PPO,
        reward_type=RewardType.SHAPED,
        model_name="stock_trading_ppo",
        num_episodes=100,
        max_steps_per_episode=1000,
        learning_rate=3e-4,
        batch_size=64,
        update_interval=2048
    )
    
    # 创建智能体
    agent = StockTradingAgent(
        config=config,
        env=env,
        reward_func=reward_func,
        callbacks=[callback]
    )
    
    # 构建模型
    await agent.build_model()
    
    # 训练模型
    print("开始训练...")
    metrics = await agent.train()
    print(f"训练完成！最终平均奖励: {metrics[-1].reward_mean:.4f}")
    
    # 保存模型
    await agent.save("./stock_trading_model")
    
    # 评估模型
    print("\n开始评估...")
    eval_metrics = await agent.evaluate(num_episodes=10)
    print(f"评估完成！平均奖励: {eval_metrics.reward_mean:.4f}")


if __name__ == "__main__":
    asyncio.run(main()) 