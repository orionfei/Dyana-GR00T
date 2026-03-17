import asyncio
import websockets
import json
import base64
import numpy as np
import cv2
from PIL import Image


class UnityServer:
    def __init__(self, host="0.0.0.0", port=8765, resize_size=(256, 256)):
        self.host = host
        self.port = port
        self.websocket = None   # 存储当前 Unity client 的连接
        self._recv_task = None
        self._obs_queue = asyncio.Queue()
        self._metrics_queue = asyncio.Queue()
        self._step_feedback_queue = asyncio.Queue()
        self._connection_event = asyncio.Event()  # 用于等待连接建立
        self.server = None
        self.resize_size = resize_size

        self.task_des_dict = {
            "circular": "Grab the object in the video that is making a circular motion",
            "linear": "Grab the object in the video that is making a straight motion",
            "harmonic": "Grab the object in the video that is doing simple harmonic motion"
        }

        self.debug_idx = 0

    async def start(self):
        """启动 WebSocket 服务，等待 Unity client 连接"""
        async def handler(websocket):
            print(f"🔌 Unity 已连接：{websocket.remote_address}")
            self.websocket = websocket
            self._connection_event.set()  # 通知连接已建立
            try:
                async for msg in websocket:
                    await self._handle_message(msg)
            except Exception as e:
                print("❌ Unity 连接异常:", e)
            finally:
                self.websocket = None
                self._connection_event.clear()  # 清除连接状态
                print("🔌 Unity 已断开连接")

        self.server = await websockets.serve(handler, self.host, self.port)
        print(f"🚀 Python WebSocket 服务已启动: ws://{self.host}:{self.port}")

    async def _handle_message(self, msg):
        """处理来自 Unity 的消息"""
        try:
            msg = json.loads(msg)
            if msg["type"] == "image_and_state":
                data = json.loads(msg["data"])

                # 处理图像
                img_bytes = base64.b64decode(data["image_data"])
                np_arr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, self.resize_size)  # resize 到256x256

                # 处理状态 - 使用 float64 保持与训练时一致
                state = np.array(data["state_data"], dtype=np.float64)  # (18,) 或 (3,)

                # 这里的obs里面每一项前面加一个维度，是为了适配GR00T的接口
                obs = {
                        "video.ego_view": np.expand_dims(image_rgb, axis=0),  # [1,H,W,C]
                        "state.left_hand": state[np.newaxis, ...],  # [1,18] 或 [1,3]
                        "annotation.human.action.task_description": np.array([data["task_type"]]),
                    }
                self._obs_queue.put_nowait(obs)
                print(f"✅ 收到 obs，已经放入队列")

                # 储存image为jpg
                # Image.fromarray(image_rgb).save(f"/mnt/sdc/bch/forBenchmark/Isaac-GR00T/debug_image/image_{data['episode_id']}_{data['repeat']}_{data['step']}_rgb.jpg")
                # Image.fromarray(image).save(f"/mnt/sdc/bch/forBenchmark/Isaac-GR00T/debug_image/image_{data['episode_id']}_{data['repeat']}_{data['step']}_bgr.jpg")
            
            elif msg["type"] == "metrics":
                data = json.loads(msg["data"])
                metrics = {
                    "episode_id": data["episode_id"],
                    "repeat": data["repeat"],
                    "success": data["success"],
                    "waitTime": data["waitTime"],
                    "score": data["score"],
                    "min_XZ": data["min_distance_to_target"],
                    "successIndex": data["successIndex"],
                    "minJointToSurfaceDistance": data["minJointToSurfaceDistance"],
                }
                self._metrics_queue.put_nowait(metrics)
                print(f"📉 收到 episode: {data['episode_id']}, repeat: {data['repeat']} 的 metrics（{data['score']}, {data['success']}），已经放入队列")

            elif msg["type"] == "step_feedback":
                data = json.loads(msg["data"])
                feedback = {
                    "episode_id": data["episode_id"],
                    "repeat": data["repeat"],
                    "decision_step": data["decision_step"],
                    "done": data["done"],
                    "success": data["success"],
                    "task_type": data["task_type"],
                    "current_frame_index": data["current_frame_index"],
                    "successIndex": data["successIndex"],
                    "min_distance_to_target": data["min_distance_to_target"],
                    "minJointToSurfaceDistance": data["minJointToSurfaceDistance"],
                }
                self._step_feedback_queue.put_nowait(feedback)
                print(
                    "🧭 收到 episode: "
                    f"{data['episode_id']}, repeat: {data['repeat']} 的 step_feedback "
                    f"(step={data['decision_step']}, done={data['done']}, success={data['success']})"
                )

            else:
                print("⚠️ 收到未知消息类型:", msg["type"])

        except Exception as e:
            print("❌ 解析 Unity 消息失败:", e)

    async def wait_for_connection(self, timeout=30):
        """等待 Unity client 连接，阻塞式"""
        print(f"⏳ 等待 Unity client 连接... (超时: {timeout}秒)")
        try:
            await asyncio.wait_for(self._connection_event.wait(), timeout=timeout)
            print("✅ Unity client 已连接")
            return True
        except asyncio.TimeoutError:
            print(f"❌ 等待 Unity client 连接超时 ({timeout}秒)")
            return False

    def wait_for_connection_sync(self, timeout=30):
        """同步版本的等待连接"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.wait_for_connection(timeout))

    def is_connected(self):
        """检查是否已连接"""
        if self.websocket is None:
            return False
        try:
            # 在 websockets 15.0.1 中，使用 state 属性检查连接状态
            # OPEN = 1, CLOSING = 2, CLOSED = 3
            if hasattr(self.websocket, 'state'):
                return self.websocket.state == 1
            # 尝试使用 closed 属性
            elif hasattr(self.websocket, 'closed'):
                return not self.websocket.closed
            # 最后尝试检查 close_code
            elif hasattr(self.websocket, 'close_code'):
                return self.websocket.close_code is None
            else:
                # 如果都没有，假设连接正常
                return True
        except Exception as e:
            print(f"⚠️ 检查连接状态时出错: {e}")
            return False

    def debug_connection_info(self):
        """调试连接信息"""
        if self.websocket is None:
            print("🔍 WebSocket: None")
            return
        
        print(f"🔍 WebSocket 对象: {type(self.websocket)}")
        for attr in ['state', 'closed', 'close_code', 'open']:
            if hasattr(self.websocket, attr):
                try:
                    value = getattr(self.websocket, attr)
                    print(f"🔍 {attr}: {value}")
                except Exception as e:
                    print(f"🔍 {attr}: Error accessing - {e}")
            else:
                print(f"🔍 {attr}: Not available")

    def get_obs(self, block=True, timeout=None):
        """获取一帧 obs"""
        loop = asyncio.get_event_loop()
        if block:
            try:
                return loop.run_until_complete(asyncio.wait_for(self._obs_queue.get(), timeout))
            except asyncio.TimeoutError:
                return None
        else:
            if self._obs_queue.empty():
                return None
            return self._obs_queue.get_nowait()
    
    def get_metrics_from_unity(self, block=None, timeout=None):
        """获取 Unity端统计 的 waitTime 和 success情况"""
        loop = asyncio.get_event_loop()
        if block:
            try:
                return loop.run_until_complete(asyncio.wait_for(self._metrics_queue.get(), timeout))
            except asyncio.TimeoutError:
                return None
        else:
            if self._metrics_queue.empty():
                return None
            return self._metrics_queue.get_nowait()

    def get_step_feedback(self, block=True, timeout=None):
        """获取 Unity 发送的 chunk 级 step feedback。"""
        loop = asyncio.get_event_loop()
        if block:
            try:
                return loop.run_until_complete(asyncio.wait_for(self._step_feedback_queue.get(), timeout))
            except asyncio.TimeoutError:
                return None
        else:
            if self._step_feedback_queue.empty():
                return None
            return self._step_feedback_queue.get_nowait()

    def get_step_feedback_sync(self, block=True, timeout=None):
        """同步版本的 get_step_feedback。"""
        return self.get_step_feedback(block=block, timeout=timeout)

    async def send_start_episode(self, episode_id, task_type, repeat_num, steps, start_frame_idx, windowSize):
        """通知 Unity 开始一个 episode"""
        if not self.is_connected():
            print("⚠️ 没有 Unity 客户端连接")
            return False
        try:
            msg = {
                "type": "start_episode",
                "data": json.dumps({"episode_id": int(episode_id), "task_type": task_type, "retry_time": int(repeat_num), "total_frame_episode": int(steps), "start_frame_idx": int(start_frame_idx), "windowSize": int(windowSize)})
            }
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ 发送 start_episode 失败: {e}")
            self.websocket = None  # 重置连接状态
            self._connection_event.clear()
            return False

    async def send_action_data(self, actions):
        """发送动作序列给 Unity"""
        if not self.is_connected():
            print("⚠️ 没有 Unity 客户端连接")
            return False
        try:
            if isinstance(actions, np.ndarray):
                actions = actions.tolist()

            # ✅ 关键改动：每一帧包一层 {"values": frame}
            wrapped_actions = [{"values": frame} for frame in actions]

            msg = {
                "type": "action_data",
                "data": json.dumps({"actions": wrapped_actions})
            }
            # print(f"DEBUG: send_action_data: {wrapped_actions[:2]} ...")  # 打印前两帧看看
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ 发送 action_data 失败: {e}")
            self.websocket = None  # 重置连接状态
            self._connection_event.clear()
            return False
        
    async def send_inference_complete(self):
        """通知 Unity 本次推理完成"""
        if not self.is_connected():
            print("⚠️ 没有 Unity 客户端连接")
            return False
        try:
            msg = {"type": "inference_complete", "data": "{}"}
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ 发送 inference_complete 失败: {e}")
            self.websocket = None  # 重置连接状态
            self._connection_event.clear()
            return False

    async def send_save_results(self):
        """通知 Unity 保存所有结果"""
        if not self.is_connected():
            print("⚠️ 没有 Unity 客户端连接")
            return False
        try:
            msg = {"type": "save_results", "data": "{}"}
            await self.websocket.send(json.dumps(msg))
            return True
        except Exception as e:
            print(f"❌ 发送 save_results 失败: {e}")
            self.websocket = None  # 重置连接状态
            self._connection_event.clear()
            return False

    # 同步版本的 send 方法，用于评估脚本
    def send_start_episode_sync(self, episode_id, task_type, repeat_num, steps, start_frame_idx, windowSize):
        """同步版本的 send_start_episode"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_start_episode(episode_id, task_type, repeat_num, steps, start_frame_idx, windowSize))

    def send_action_data_sync(self, actions):
        """同步版本的 send_action_data"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_action_data(actions))

    def send_inference_complete_sync(self):
        """同步版本的 send_inference_complete"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_inference_complete())

    def send_save_results_sync(self):
        """同步版本的 send_save_results"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.send_save_results())

    async def stop(self):
        """关闭服务"""
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        print("🛑 Python WebSocket 服务已关闭")
