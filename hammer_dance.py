"""
ハンマー＆ダンス戦略を用いた感染症数理モデルシミュレーション

目的：ハンマー＆ダンスを再現する数理モデルを構築し，
      感染症対策や流行動態の解明に役立てる

SIRモデル微分方程式:
S' = -βSI
I' = βSI - γI  
R' = γI

ここで：
S: 感受性者数 (Susceptible)
I: 感染者数 (Infected)
R: 回復者数 (Recovered)
β: 感染率
γ: 回復率
"""

import math

class HammerDanceModel:
    def __init__(self, N, I0, R0, beta_hammer, beta_dance, gamma):
        """
        ハンマー＆ダンスモデルの初期化
        
        Parameters:
        N: 総人口
        I0: 初期感染者数
        R0: 初期回復者数
        beta_hammer: ハンマー期間の感染率（低い値）
        beta_dance: ダンス期間の感染率（高い値）
        gamma: 回復率
        """
        self.N = N
        self.S0 = N - I0 - R0
        self.I0 = I0
        self.R0 = R0
        self.beta_hammer = beta_hammer
        self.beta_dance = beta_dance
        self.gamma = gamma
        
    def sir_derivatives(self, S, I, R, beta):
        """
        SIRモデルの微分方程式（正規化版）
        
        s' = -βsi (s = S/N, i = I/N, r = R/N)
        i' = βsi - γi
        r' = γi
        """
        # 正規化された変数での計算
        s = S / self.N
        i = I / self.N
        r = R / self.N
        
        dsdt = -beta * s * i
        didt = beta * s * i - self.gamma * i
        drdt = self.gamma * i
        
        # 実数値に戻す
        dSdt = dsdt * self.N
        dIdt = didt * self.N
        dRdt = drdt * self.N
        
        return dSdt, dIdt, dRdt
    
    def euler_step(self, S, I, R, beta, dt):
        """
        オイラー法による1ステップ計算
        """
        dSdt, dIdt, dRdt = self.sir_derivatives(S, I, R, beta)
        S_new = S + dSdt * dt
        I_new = I + dIdt * dt
        R_new = R + dRdt * dt
        return S_new, I_new, R_new
    
    def simulate_dynamic_hammer_dance(self, max_days, hammer_threshold, dance_threshold,
                                     hammer_duration, dance_duration, dt):
        """
        動的ハンマー＆ダンス戦略のシミュレーション
        感染者数が閾値を超えたときにハンマー期間を開始
        感染者数が下限閾値を下回ったときにダンス期間に切り替え
        
        Parameters:
        max_days: 最大シミュレーション日数
        hammer_threshold: ハンマー期間開始の感染者数閾値（上限）
        dance_threshold: ダンス期間切り替えの感染者数閾値（下限）
        hammer_duration: ハンマー期間の最大日数
        dance_duration: ダンス期間の日数（参考値）
        dt: 時間刻み
        """
        steps = int(max_days / dt)
        
        # 結果を格納するリスト
        t_list = []
        S_list = []
        I_list = []
        R_list = []
        phase_list = []
        transition_points = []  # フェーズ切り替えポイント
        
        # 初期条件
        S, I, R = self.S0, self.I0, self.R0
        t = 0
        
        # 初期状態（ダンス期間から開始）
        current_phase = 'dance'
        phase_start_time = 0
        phase_duration = 0
        
        for step in range(steps):
            t_list.append(t)
            S_list.append(S)
            I_list.append(I)
            R_list.append(R)
            phase_list.append(current_phase)
            
            # フェーズ切り替えの判定
            if current_phase == 'dance':
                # ダンス期間中：感染者数が上限閾値を超えたらハンマー期間に切り替え
                if I > hammer_threshold:
                    current_phase = 'hammer'
                    phase_start_time = t
                    transition_points.append((t, 'dance_to_hammer', I))
                    print(f"時刻 {t:.1f}日: 感染者数 {I:.1f}人 -> ハンマー期間開始")
                
                # ダンス期間で微分方程式を解く
                S, I, R = self.euler_step(S, I, R, self.beta_dance, dt)
                
            elif current_phase == 'hammer':
                # ハンマー期間中の切り替え判定
                
                # 感染者数が下限閾値を下回った場合は即座にダンス期間に切り替え
                if I < dance_threshold:
                    current_phase = 'dance'
                    phase_start_time = t
                    transition_points.append((t, 'hammer_to_dance_threshold', I))
                    print(f"時刻 {t:.1f}日: 感染者数 {I:.1f}人 -> ダンス期間切り替え（閾値到達）")
                
                # ハンマー期間で微分方程式を解く
                S, I, R = self.euler_step(S, I, R, self.beta_hammer, dt)
            
            t += dt
            
            # 感染者がほぼ0になったら終了
            if I < 0.1:
                print(f"時刻 {t:.1f}日: 感染終息 (感染者数: {I:.1f}人)")
                break
        
        return t_list, S_list, I_list, R_list, phase_list, transition_points
    
    def print_results(self, t, S, I, R, phases, hammer_days=30, dance_days=20):
        """
        シミュレーション結果の表示（テキスト形式）
        """
        print("\n=== ハンマー＆ダンス戦略シミュレーション結果 ===")
        print(f"総人口: {self.N}人")
        print(f"初期感染者: {self.I0}人")
        print(f"ハンマー期間感染率: {self.beta_hammer}")
        print(f"ダンス期間感染率: {self.beta_dance}")
        print(f"回復率: {self.gamma}")
        print()
        
        # 各サイクルの結果表示
        cycle_length = hammer_days + dance_days
        cycle_steps = int(cycle_length / 0.1)  # dt=0.1を仮定
        
        for cycle in range(len(t) // cycle_steps):
            start_idx = cycle * cycle_steps
            end_idx = min((cycle + 1) * cycle_steps, len(t))
            
            if end_idx <= len(I):
                cycle_max_I = max(I[start_idx:end_idx])
                cycle_end_I = I[end_idx-1] if end_idx-1 < len(I) else I[-1]
                
                print(f"サイクル {cycle+1}:")
                print(f"  最大感染者数: {cycle_max_I:.1f}人")
                print(f"  サイクル終了時感染者数: {cycle_end_I:.1f}人")
        
        # 全体の統計
        max_I = max(I)
        max_I_day = t[I.index(max_I)]
        final_S = S[-1]
        final_I = I[-1]
        final_R = R[-1]
        
        # 正規化された値（比率）
        max_I_ratio = max_I / self.N
        final_S_ratio = final_S / self.N
        final_I_ratio = final_I / self.N
        final_R_ratio = final_R / self.N
        
        print(f"\n全体統計:")
        print(f"最大感染者数: {max_I:.1f}人 ({max_I_ratio:.4f})")
        print(f"ピーク到達日: {max_I_day:.1f}日")
        print(f"最終状態:")
        print(f"  感受性者: {final_S:.1f}人 ({final_S_ratio:.4f})")
        print(f"  感染者: {final_I:.1f}人 ({final_I_ratio:.6f})")
        print(f"  回復者: {final_R:.1f}人 ({final_R_ratio:.4f})")
        print(f"総感染率: {final_R_ratio * 100:.2f}%")
    
    def save_csv_results(self, t, S, I, R, phases, filename="hammer_dance_results.csv"):
        """
        結果をCSVファイルに保存（正規化版も含む）
        """
        try:
            with open(filename, 'w') as f:
                f.write("Time,Susceptible,Infected,Recovered,Phase,S_ratio,I_ratio,R_ratio\n")
                for i in range(len(t)):
                    s_ratio = S[i] / self.N
                    i_ratio = I[i] / self.N
                    r_ratio = R[i] / self.N
                    f.write(f"{t[i]:.1f},{S[i]:.1f},{I[i]:.1f},{R[i]:.1f},{phases[i]},{s_ratio:.6f},{i_ratio:.6f},{r_ratio:.6f}\n")
            print(f"\n結果を {filename} に保存しました")
        except Exception as e:
            print(f"CSVファイル保存中にエラーが発生しました: {e}")
    
    def plot_text_graph(self, t, S, I, R, phases, width=80, height=20):
        """
        テキストベースのグラフ表示
        """
        print("\n=== 感染者数推移グラフ（テキスト版） ===")
        
        # データの正規化
        max_I = max(I)
        min_I = min(I)
        
        # グラフの範囲調整
        if max_I == min_I:
            normalized_I = [height // 2] * len(I)
        else:
            normalized_I = [int((i - min_I) / (max_I - min_I) * (height - 1)) for i in I]
        
        # 時間軸の調整（表示する点数を制限）
        step = max(1, len(t) // width)
        display_t = t[::step]
        display_I = normalized_I[::step]
        display_phases = phases[::step]
        
        # グラフの描画
        for row in range(height - 1, -1, -1):
            line = ""
            for col, (time, norm_i, phase) in enumerate(zip(display_t, display_I, display_phases)):
                if norm_i == row:
                    if phase == 'hammer':
                        line += "H"  # ハンマー期間
                    else:
                        line += "D"  # ダンス期間
                elif norm_i > row:
                    line += "|"
                else:
                    line += " "
            
            # Y軸ラベル
            y_value = min_I + (max_I - min_I) * row / (height - 1)
            print(f"{y_value:6.1f} |{line}")
        
        # X軸
        print("       " + "-" * len(display_t))
        
        # 時間軸ラベル
        time_labels = ""
        for i, time in enumerate(display_t):
            if i % 10 == 0:  # 10ステップごとにラベル表示
                label = f"{time:3.0f}"
                time_labels += label + " " * (10 - len(label))
        print(f"       {time_labels}")
        print("       時間（日）")
        
        print("\n凡例: H=ハンマー期間, D=ダンス期間")
        print(f"感染者数範囲: {min_I:.1f} - {max_I:.1f}人")
    
    def create_html_graph(self, t, S, I, R, phases, hammer_threshold, dance_threshold, filename="hammer_dance_graph.html"):
        """
        HTML形式のグラフを生成
        """
        try:
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>動的制御ハンマー＆ダンス戦略シミュレーション結果</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .graph-container {{ margin: 20px 0; }}
        .stats {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .control-info {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>動的制御ハンマー＆ダンス戦略による感染症シミュレーション</h1>
    
    <div class="control-info">
        <h3>動的制御ルール</h3>
        <p>🔨 <strong>ハンマー期間開始</strong>: 感染者数が{hammer_threshold}人を超えたとき</p>
        <p>💃 <strong>ダンス期間切り替え</strong>: 感染者数が{dance_threshold}人を下回ったとき（または最大30日経過）</p>
    </div>
    
    <div class="stats">
        <h3>パラメータ設定</h3>
        <p>総人口: {self.N}人 | 初期感染者: {self.I0}人 | ハンマー期間感染率: {self.beta_hammer} | ダンス期間感染率: {self.beta_dance} | 回復率: {self.gamma}</p>
    </div>
    
    <div class="graph-container">
        <div id="sirGraph" style="width:100%;height:500px;"></div>
    </div>
    
    <div class="graph-container">
        <div id="infectedGraph" style="width:100%;height:400px;"></div>
    </div>
    
    <script>
        // SIRモデル全体のグラフ
        var trace1 = {{
            x: {t},
            y: {S},
            type: 'scatter',
            mode: 'lines',
            name: '感受性者 (S)',
            line: {{color: 'blue'}}
        }};
        
        var trace2 = {{
            x: {t},
            y: {I},
            type: 'scatter',
            mode: 'lines',
            name: '感染者 (I)',
            line: {{color: 'red'}}
        }};
        
        var trace3 = {{
            x: {t},
            y: {R},
            type: 'scatter',
            mode: 'lines',
            name: '回復者 (R)',
            line: {{color: 'green'}}
        }};
        
        var layout1 = {{
            title: 'SIRモデル - 動的制御ハンマー＆ダンス戦略',
            xaxis: {{ title: '時間 (日)' }},
            yaxis: {{ title: '人数' }},
            showlegend: true
        }};
        
        Plotly.newPlot('sirGraph', [trace1, trace2, trace3], layout1);
        
        // 感染者数詳細グラフ（閾値ライン付き）
        var trace4 = {{
            x: {t},
            y: {I},
            type: 'scatter',
            mode: 'lines',
            name: '感染者数',
            line: {{color: 'red', width: 3}}
        }};
        
        // ハンマー開始閾値ライン
        var trace5 = {{
            x: [0, Math.max(...{t})],
            y: [{hammer_threshold}, {hammer_threshold}],
            type: 'scatter',
            mode: 'lines',
            name: 'ハンマー開始閾値 ({hammer_threshold}人)',
            line: {{color: 'orange', width: 2, dash: 'dash'}}
        }};
        
        // ダンス切り替え閾値ライン
        var trace6 = {{
            x: [0, Math.max(...{t})],
            y: [{dance_threshold}, {dance_threshold}],
            type: 'scatter',
            mode: 'lines',
            name: 'ダンス切り替え閾値 ({dance_threshold}人)',
            line: {{color: 'lightblue', width: 2, dash: 'dot'}}
        }};
        
        var layout2 = {{
            title: '感染者数の推移（動的制御ハンマー＆ダンス戦略）',
            xaxis: {{ title: '時間 (日)' }},
            yaxis: {{ title: '感染者数' }},
            showlegend: true
        }};
        
        Plotly.newPlot('infectedGraph', [trace4, trace5, trace6], layout2);
    </script>
    
    <div class="stats">
        <h3>シミュレーション結果サマリー</h3>
        <p>最大感染者数: {max(I):.1f}人</p>
        <p>最終感受性者: {S[-1]:.1f}人</p>
        <p>最終感染者: {I[-1]:.1f}人</p>
        <p>最終回復者: {R[-1]:.1f}人</p>
        <p>総感染率: {((self.N - S[-1]) / self.N * 100):.1f}%</p>
        <p>シミュレーション期間: {t[-1]:.1f}日</p>
    </div>
</body>
</html>"""
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\nHTMLグラフを {filename} に保存しました")
            print("ブラウザで開いてグラフを表示できます")
            
        except Exception as e:
            print(f"HTMLファイル作成中にエラーが発生しました: {e}")
    
    def create_excel_data(self, t, S, I, R, phases, filename="hammer_dance_data.txt"):
        """
        Excel用のタブ区切りデータを生成
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("時間\t感受性者\t感染者\t回復者\tフェーズ\n")
                for i in range(len(t)):
                    f.write(f"{t[i]:.1f}\t{S[i]:.1f}\t{I[i]:.1f}\t{R[i]:.1f}\t{phases[i]}\n")
            print(f"\nExcel用データを {filename} に保存しました")
            print("このファイルをExcelで開いてグラフを作成できます")
            
        except Exception as e:
            print(f"Excel用ファイル作成中にエラーが発生しました: {e}")

def main():
    """
    メイン実行関数
    """
    # 設定値を変数として定義
    hammer_threshold_val = 194323
    dance_threshold_val = 10000
    
    print("ハンマー＆ダンス戦略シミュレーション開始...")
    print(f"動的制御：感染者数が{hammer_threshold_val}人を超えたときにハンマー期間開始")
    print(f"　　　　　感染者数が{dance_threshold_val}人を下回ったときにダンス期間切り替え")
    
    # モデルのパラメータ設定
    model = HammerDanceModel(
        N=125000000,           # 総人口
        I0=194323,            # 初期感染者数
        R0=20000000,             # 初期回復者数
        beta_hammer=0.142857,  # ハンマー期間の感染率（厳格な対策）
        beta_dance=1.428571,   # ダンス期間の感染率（緩和された対策）
        gamma=0.142857         # 回復率
    )
    
    # 動的ハンマー＆ダンス戦略のシミュレーション実行
    print("\n=== 動的ハンマー＆ダンス戦略 ===")
    t, S, I, R, phases, transitions = model.simulate_dynamic_hammer_dance(
        max_days=1095,                    # 最大1095日
        hammer_threshold=hammer_threshold_val,  # 感染者数でハンマー期間開始
        dance_threshold=dance_threshold_val,    # 感染者数でダンス期間切り替え
        hammer_duration=None,            # ハンマー期間：無制限（感染者数が閾値を下回るまで継続）
        dance_duration=None,             # ダンス期間：無制限（感染者数が閾値を超えるまで継続）
        dt=0.1                          # 時間刻み
    )
    
    # フェーズ切り替えポイントの表示
    print(f"\n=== フェーズ切り替え履歴 ===")
    for time, transition_type, infected_count in transitions:
        if transition_type == 'dance_to_hammer':
            print(f"{time:.1f}日: ダンス→ハンマー (感染者数: {infected_count:.1f}人)")
        elif transition_type == 'hammer_to_dance_threshold':
            print(f"{time:.1f}日: ハンマー→ダンス【閾値到達】 (感染者数: {infected_count:.1f}人)")
        elif transition_type == 'hammer_to_dance_timeout':
            print(f"{time:.1f}日: ハンマー→ダンス【期間満了】 (感染者数: {infected_count:.1f}人)")
    
    # 結果の表示
    model.print_results(t, S, I, R, phases)
    
    # テキストベースのグラフ表示
    model.plot_text_graph(t, S, I, R, phases)
    
    # CSV形式で結果保存
    model.save_csv_results(t, S, I, R, phases, "dynamic_hammer_dance_results.csv")
    
    # HTMLグラフの生成（動的版）
    model.create_html_graph(t, S, I, R, phases, hammer_threshold_val, dance_threshold_val, "dynamic_hammer_dance_graph.html")
    
    # Excel用データの生成
    model.create_excel_data(t, S, I, R, phases, "dynamic_hammer_dance_data.txt")
    
    # サンプルデータの表示（最初の50データポイント）
    print("\n=== サンプルデータ（最初の50データポイント） ===")
    print("時刻\t感受性者\t\t感染者\t\t回復者\t\tフェーズ")
    print("(日)\t(人数/比率)\t\t(人数/比率)\t\t(人数/比率)")
    for i in range(0, min(50, len(t)), 5):  # 5ステップごとに表示
        s_ratio = S[i] / model.N
        i_ratio = I[i] / model.N
        r_ratio = R[i] / model.N
        print(f"{t[i]:.1f}\t{S[i]:.1f}({s_ratio:.4f})\t{I[i]:.1f}({i_ratio:.6f})\t{R[i]:.1f}({r_ratio:.4f})\t{phases[i]}")

if __name__ == "__main__":
    main()
