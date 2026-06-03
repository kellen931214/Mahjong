你現在是一位資深的 AI 遊戲演算法專家。我目前正在開發一個基於 Decision Mamba  的日本麻將 AI 專案，我想參考微軟亞洲研究院發表的減量模型「Suphx」論文中的評估指標體系，來為我的模型編寫一套「評估與統計模組（Evaluation & Metrics Module）」。

請根據以下從 Suphx 論文中提取的指標規範，為我產出完整的 Python 程式碼實作：

1. 離線模型準確率 (Offline Action Accuracy)：
   - 請寫一個函式，傳入模型的預測 logits 與真實行為標籤 target_action（包含遮罩），能夠分別計算並輸出：
     * 純切牌準確率 (Dahai Accuracy, 動作 ID 0~73)
     * 吃牌準確率 (Chow Accuracy, 動作 ID 74~103)
     * 碰牌準確率 (Pong Accuracy, 動作 ID 104~140)
     * 槓牌準確率 (Kong Accuracy, 動作 ID 141~174)
     * 立直準確率 (Riichi Accuracy, 動作 ID 177)

2. 線上對局統計器 (Online Tracker Class)：
   - 請實作一個 MahjongMetricTracker 類別，用來追蹤模型在數千場自主對弈（Self-Play），需動態維護並計算以下指標：
     * 和了率 (Win Rate)：自己胡牌的次數 / 總局數
     * 放銃率 (Deal-in Rate)：點炮給對手的次數 / 總局數
     * 順位分佈 (Placement Distribution)：統計獲得 1、2、3、4 名的精確百分比
     * 讓我看到這局是和局還是有贏牌


寫一個evaluate.py 和 存放指標函式的py 跑evaluate.py的時候 分成兩個mode我可以自由選擇要離線的或自我對弈的指標