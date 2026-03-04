---
tags : 
  - type:article
  - topic:paper-reading
state: draft

---

# idea
<!-- 文章にする前の気づき、input情報など雑記.あまり変更しない -->
on policy self distillationは RL, SFTと並ぶLLM学習の基本的な学習パラダイムとなる重要な要素になりそう。

比較的新しく、 on-policy distillation,　self-distillationの関連研究がまとまっており、適切なablation studyをされている良論文だと感じたため、シンプルに概要をまとめ、AI研究者としての気になる点をまとめて文章とする。

- https://arxiv.org/html/2601.18734v1

良い点
-  on-policy distillation,　self-distillationの関連研究がまとまっており
- OPSDの良さがわかりやすく実験で検証されている
- Full-vocabulary divergenceと　シンプルなPolicy gradient設定の比較がされている

気になりポイント
- sftモデル自体元々ある程度性能が高く、GRPO、OPSDでのゲインが少ないので、これだけだと素直に良いとは言い切れないかも。続きの検証が求められる
- response lengthについて GRPOは16kを用いているが, OPSDでは最大4kしか使っていない。これはreasoning前半のみを学習に使っていることになり、そのバイアスが良い影響を生み出している可能性があるのでは？（むしろ16kで学習した時にうまくいかない可能性はないか？）
- betaの値による変化はある？

まとめポイント

土台となる技術
- self-distillation
- on-policy distillation

実験概要
- model: Qwen3
- data: OpenThought
- 評価：aimeなどmath系のタスク


# Draft
<!-- ユーザーによる文章案のscrachpad.あまり変更しない -->


# for AI Draft
<!-- AI向けの取材用scratchpad draft, idea以外に調査したり、ユーザーに深掘り質問した内容を記載質問した内容を記載 -->


# Main
<!-- 記事本文。この記事を書くプラットフォーム向けにローカライズするため、やや冗長さは情報は削りすぎずない良い文章を目指す -->

この記事は、論文「Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models」（arXiv:2601.18734v1）を読んで、On-Policy Self-Distillation（OPSD）の要点と、個人的に気になった点をまとめたメモです。  
論文URL: https://arxiv.org/html/2601.18734v1

## まず結論：OPSDは「教師なし」ではなく「同一モデル内の教師化」

最近の推論モデルのポストトレーニングは、ざっくり言うと SFT（高品質な推論トレースに合わせる）と RLVR（正誤が検証できる報酬で強化学習）と distillation の三つ巴になってきました。  
その中で on-policy distillation は「生徒が自分でサンプルした出力（on-policy）に対して、教師がトークンごとに濃い信号を返す」ので、RL の疎な報酬よりも効率がよい、という流れがあります。

ただ、従来の on-policy distillation は「別の教師モデル」が前提になりがちです。ここに対してこの論文は、**同一モデルが“条件付け”の違いだけで teacher / student を兼ねる**という発想で、OPSD を提案しています。  
teacher は「問題 + 正解の推論トレース（参照解）」を見られる（privileged information）一方、student は「問題だけ」を見て解く。student が自分のロールアウトを生成し、そのロールアウト上で teacher と student の次トークン分布のズレを最小化する、という設計です。

![OPSDの全体像（論文 Figure 1）](20260222-on-policy-self-distillation-paper-reading/assets/opsd-overview-fig1.png)

## 背景：SFT / GRPO の弱点を“トークンごとの教師信号”で埋めたい

論文の問題設定はかなり素直で、既存手法の弱点が並びます。

- **SFT（オフポリシー distillation）**は、学習時に見た prefix と推論時の prefix がずれてしまう（distribution mismatch）ので、生成の自己回帰の途中で破綻しやすい、という古典的な話がある。
- **RLVR（例：GRPO）**は、最終的な正解/不正解のような「疎な報酬」しかないので、どのトークンが悪かったのかを直接教えてくれない。また、GRPO は各プロンプトで複数ロールアウトを必要とし（この論文の設定では 8 本）、計算・サンプルコストが重い。

OPSD はここを、**参照解を使って teacher を“密”にする**ことで埋めにいきます。ポイントは「正解があるタスク（数学など）では、参照解という“特権情報”を使って teacher 側の次トークン分布をより情報豊かにできる」ことです。

## 手法：teacher は「(x, y*)」、student は「x」だけを見る

OPSD の形式化はシンプルで、同じモデルから 2 つの条件付き分布を作ります。

- student policy: `p_S(· | x)`（問題文のみ）
- teacher policy: `p_T(· | x, y*)`（問題文 + 参照解）

学習の流れは次の通りです。

1. student が `x` だけを見て、解答 `ŷ` をサンプルする（on-policy rollout）
2. 各ステップで、同じ prefix `ŷ_<n` に対して teacher と student の次トークン分布を計算する  
   - student は `(x, ŷ_<n)` に条件付け  
   - teacher は `(x, y*, ŷ_<n)` に条件付け（参照解が入るぶん“よく分かっている”）
3. teacher と student の分布の距離（divergence）を、トークン位置ごとに最小化する

実装上の重要な工夫として、論文では **teacher を“更新中のモデル”ではなく、初期モデルに固定**しています（student だけを更新する）。これが学習を安定させ、初期モデルからの過度な逸脱を抑える正則化にもなっている、という位置づけです。

### 学習目的：full-vocabulary の divergence（ログイット distillation）

OPSD は各トークン位置で、teacher と student の「語彙全体の分布」を使って divergence を計算します（full softmax を使うタイプ）。論文の実験では divergence として `JSD_{β=0.5}` を使っています。  
直感的には「teacher が参照解を読んだ上で“次に来るトークンの分布”を出し、それに student を合わせる」ので、RL のように一発の正誤報酬を返すよりも、どこで何を間違えたかの密度が高い信号になります。

### 代替案：sampled-token（policy gradient 風）

比較として、student が実際にサンプルしたトークンだけを評価して、teacher / student の logprob の差を advantage のように使う variant も検討されています。結果は後述しますが、full-vocabulary のほうが一貫して強い（ただしメモリは重い）という話になっています。

## 実験設定と主要結果（Qwen3 + OpenThoughts + 数学ベンチ）

実験は Qwen3 の instruct（1.7B / 4B / 8B）をベースに、OpenThoughts の数学推論サブセット（最大 30K）で学習し、AIME 2024/2025, HMMT 2025, AMO-Bench で評価しています。  
比較は Base / +SFT / +GRPO / +OPSD。ここが読みやすかったポイントで、まず「同じデータを使った時に、SFT と GRPO と OPSD がどう並ぶか」が正面から出ています。

平均スコア（Average@16）のみ抜き出すと、Table 2 は次の通りです。
（評価のサンプリング設定は Qwen3 の推奨パラメータに従い、temperature 1.2 / generation length 38k などで average@16 を報告しています。）

| Model      | Base | +SFT | +GRPO | +OPSD |
| ---------- | ---: | ---: | ----: | ----: |
| Qwen3-1.7B | 28.8 | 28.0 |  30.5 |  30.4 |
| Qwen3-4B   | 48.3 | 49.6 |  49.6 |  50.6 |
| Qwen3-8B   | 50.0 | 50.0 |  51.3 |  52.2 |

この表だけ見ると、OPSD の立ち位置はわりと明確で、

- 4B/8B では GRPO と同等以上、SFT より強い
- 1.7B では GRPO が僅差で上（ただし OPSD も Base/SFT は上回る）

という傾向です。論文中でも「self-distillation は一定以上のモデル能力が必要」という議論があり、1.7B の伸びが控えめなのは整合的に見えます。

## トークン効率：GRPO の 4–8× 省トークン（ただし“長さ”の取り扱いは要注意）

この論文の売りのひとつが token efficiency で、Figure 3 では、同じ effective batch size で OPSD と GRPO を比べると **4–8× token-efficient** だと主張しています。  
設定として、GRPO は 8 rollout / generation cap 16k、OPSD は 1 rollout / cap 2k（別の箇所では 4k も）です。

ここは個人的にいちばん重要な読みどころで、単に「rollout 本数が違う」だけでなく、**生成長の上限が異なる**ので、学習信号の質（特に reasoning の後半や検算パート）にバイアスが入る可能性があります。  
論文は generation length のアブレーション（1k/2k/4k）も用意していて、長いほど良いという結論ではありますが、「GRPO と OPSD の比較」と「OPSD 内の長さアブレーション」は、見たい問いが少し違う、という感覚も残りました。

![OPSD vs GRPO のトークン効率（論文 Figure 3）](20260222-on-policy-self-distillation-paper-reading/assets/opsd-fig3-comparison-opsd-with-grpo-by-step-and-gen-token.png)

## アブレーションで分かったこと

### 1) モデルサイズ依存

上でも触れた通り、OPSD は teacher が参照解を“理解して”次トークン分布を出すことに依存します。モデルが小さいと、その teacher が弱くなり、self-distillation が効きづらい。  
Table 2 の 1.7B の結果は、まさにその示唆として読めます。

### 2) 生成長（1k/2k/4k）：長いほど teacher 信号が増える

OPSD はロールアウト上でトークンごとの divergence を積むので、生成するトークン数自体が学習信号量を決めます。Figure 4 では 1k→2k→4k と伸ばすと pass@K が上がる、という結果が出ています。  
直感的にも、短い生成だと「途中までしか考えてない」ロールアウトが増えるので、teacher と student の比較が浅いところで終わってしまう、ということだと思います。

### 3) full-vocabulary vs sampled-token：full-vocabulary が強い（ただし重い）

Table 3 はかなりクリアで、Qwen3-4B / 2048 tokens で

- full-vocabulary: AIME25 84.1, HMMT25 60.0
- sampled-token: AIME25 82.1, HMMT25 57.3

と差が出ています。一方で、full-vocabulary は各位置で語彙全体のログイットを扱うため、ピークメモリが増える（＝実装・運用上のコストが上がる）というトレードオフも書かれています。

## 読みながら気になった点（自分用メモ）

ここからは「この論文がダメ」というより、次に見たい実験のメモです。

1つ目は、ベースの SFT モデルがすでに強いので、4B/8B でのゲインが絶対値としては大きくない点です。もちろん「上振れ余地が小さい領域」で改善できていること自体は価値ですが、実運用の意思決定としては、学習コスト（logit distillation の重さ）と釣り合うかはケースバイケースになりそうです。

2つ目は、上でも触れた response length の扱いです。Figure 3 の比較では GRPO 側は 16k、OPSD 側は 2k（別設定で 4k）という差があり、reasoning のどの部分が学習に乗るかが変わり得ます。  
もし「前半だけ学習する」ことが偶然うまく働いているなら、同じ長さ制約を揃えた場合にどうなるか、あるいは OPSD を 16k に伸ばした時に安定するのか、は気になります。

3つ目は、GRPO の式に出てくる `β`（KL penalty）や、OPSD の divergence の重み付け（JSD の β など）周りの感度です。本文では `JSD_{β=0.5}` が固定で、ここを振った時の挙動がどれくらい変わるのかは、実装したくなるポイントでした。

## まとめ：OPSDは「正解がある推論タスク」で効く、次の基本パラダイム候補

OPSD は、SFT と RL の“間”にあるようで、実際には「参照解を使った、on-policy な token-level distillation」として筋がいい設計に見えました。  
特に、別教師モデルを要求せず（同一モデルの条件違いで teacher を作る）、かつ RLVR より密な信号で学べる、というのは実装・運用の観点でも魅力があります。

一方で、full-vocabulary の計算コストや、生成長の扱い、モデルサイズの下限など、使いどころの見極めも必要そうです。自分としては、まずは「正解検証ができて参照解がある」領域（数学、定理証明、限定ドメインのプログラム合成など）で、SFT→OPSD の追加学習がどれくらい効くかを触ってみたくなりました。


# title options
mainの内容ともとにタイトル案を作成する

- On-Policy Self-Distillation（OPSD）論文メモ：同一モデルが自分を教師にする
- Self-Distilled Reasoner を読む：SFTとGRPOの“間”を埋める学習法
- OPSDはなぜ効くのか：参照解を使ったトークンレベル自己蒸留
- 「別教師なし」on-policy distillation：OPSDの設計と気になった点
- 推論モデルの次の基本技術？OPSDの要点と実験結果まとめ
