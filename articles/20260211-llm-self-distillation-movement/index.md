---
tags : 
  - type:article
  - topic:llm
  - topic:self-distillation
  - topic:post-training
state: draft

---

![OPSDのアイキャッチ](../assets/eyecatch.png)

2026年1月から2月にかけて、LLM のポストトレーニング周辺では on-policy self-distillation（OPSD）関連の研究が続けて出てきました。[Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734)、[Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)、[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)、[π-Distill](https://arxiv.org/abs/2602.04942) のような論文や解説が並んでいて、self-distillation や on-policy distillation が単発のテクニックではなく、ひとつの流れとして見え始めています。

この記事では、この OPSD とは何か、self-distillation と on-policy distillation の何が組み合わさっているのか、なぜ今重要なのか、SFT や RL とどう違うのか、そして reasoning 以外にどこまで使えそうかを整理します。自分としてはこの動きを、単なる「蒸留の派生研究」ではなく、「追加のコンテキストや言語的な情報によって教師信号を密に作り、それを使って学習する」新しい学習パラダイムとして捉えています。

## Self-DistillationとOn-Policy Distillation

この流れを理解するには、on-policy self-distillation を最初から1つの固まりとして見るより、そこに含まれている self-distillation と on-policy distillation という2つの要素を分けて見るほうが分かりやすいです。最近の議論ではこれらが近い文脈で語られますが、もともとは別の軸の概念です。

まず self-distillation は、teacher と student が別々のモデルである必要はなく、同じモデルの中で teacher 的な役割と student 的な役割を作って学ぶ、という軸です。teacher にだけ追加コンテキストや参照解、自然言語フィードバック、privileged information を見せる。あるいは同一モデルの別条件付けを teacher とみなす。重要なのは、「他の強いモデルから知識を移す」ことよりも、「自分自身を教師化して性能を上げる」ことです。

一方で on-policy distillation は、teacher が誰かという話ではなく、「どの軌跡の上で学ぶか」の軸です。従来の distillation は固定データの正解列や teacher trajectory を使う、off-policy な形で行われることが多く、それ自体は有効でも、推論時にモデルが本当に辿る状態分布とのズレが残りやすかった。そのため、期待したほど性能が伸びない場面も少なくなかったと思います。これに対して on-policy distillation では、student が今の方策で実際に生成した rollout の上で teacher 信号を返して学ぶ。つまり、モデルが本当に踏む状態分布の上で distillation を行う、という考え方です。こちらの軸の本質は、teacher が self か他モデルかではなく、「現在の方策の出力に追随して学習する」ことにあります。

## On-Policy Self-Distillationとは何か

on-policy self-distillation は、ここまで見てきた self-distillation と on-policy distillation を組み合わせた枠組みです。別教師モデルを使った on-policy distillation もありえますし、同一モデルを teacher / student に分けた self-distillation を off-policy にやることもありえます。その中で、最近よく見かける形が、同一モデルをベースに teacher を作りつつ、student 自身の rollout の上で学ぶ on-policy self-distillation です。

枠組みはシンプルで、student は通常どおり自分で出力を生成し、その生成途中の各 prefix に対して、teacher 側がより多くの情報を見たうえで密な教師信号を返します。teacher に与える追加情報としては、参照解、golden trajectory、補助コンテキスト、自然言語フィードバック、privileged information などが考えられます。学習は固定データの正解列の上ではなく、student が実際に生成した rollout の上で進むので、推論時の状態分布に近いところで更新できます。

![OPSDの概要図（Self-Distilled Reasoner, Figure 1）](../assets/opsd-figure1-overview.png)

この形の嬉しさは、使えるデータの幅が広いことです。高品質な完全教師データだけでなく、参照解つきの推論データ、answer-only だとそのまま SFT しづらいデータ、自然言語での修正指示、コンテキストの中に埋め込まれた振る舞いの手本なども、teacher 側の追加情報として活用しやすい。つまり、「正解をそのまま書き写させる」のではなく、「追加情報を使って出力の採点をしてあげる」という形で学習に乗せられます。

OPSDの狙いは、単なる圧縮ではなく性能改善です。reasoning の改善、知識獲得、継続学習での忘却抑制、モデルの振る舞いやスタイルの移植、text feedback を使った修正学習など、用途の幅が広いです。LLM は最終回答だけでなく中間の言語的推論や自己修正のプロセス自体が性能に効きやすいので、teacher 側に少しだけ多い情報を持たせて密な信号を返すこの設計は、相性が良さそうに見えています。

## なぜ今OPSDが出て来たのか

背景には、SFT と RL がそれぞれ抱える課題感が見えてきたことがあります。SFT は基本的に off-policy です。学習時には正しい教師 prefix を見ながら次トークンを学ぶ一方、推論時にはモデル自身が作った prefix に条件付けされるので、少しずれたあとにどう立て直すかまでは十分に学べません。

この点で、GRPO のような RL系の手法が注目されてきました。モデル自身の出力を使って on-policy に学習できるので、SFT よりも推論時の実際の振る舞いに近い状態で更新できます。さらに、学習の中で探索してうまくいった軌跡を強化できるので、固定データをなぞるだけでは届きにくい改善が出やすい。reasoningモデルで RL が強く効いているのは、この on-policy 性と探索の効果が大きいからです。

一方、GRPO のような RL 系は、最終的な正誤や sequence-wise な報酬に依存しやすく、出力の中の信用割当が疎です。答えが間違っていたとしても、どのトークンやどの推論ステップが悪かったのかは直接は分からない。PPO 系は価値関数を推定することでこの問題に取り組まれてきましたが、LLMが得ような問題はこの価値推定が容易ではなく、その結果省かれてきました。最近 OPSD が盛り上がっている一因は、この隙間を埋める設計が増えてきたからだと思っています。

## OPSD は distillation だが、目的は圧縮ではない

注意したいのが distillation という言葉です。distillation は圧縮の文脈で語られることが多いですが、この記事で扱う OPSD 系の主目的は圧縮ではありません。

最近の OPSD 系でやっているのは、「同じモデルをより賢くするための蒸留」です。teacher と student が別モデルである必要すらなく、同一モデルが条件付けの違いだけで teacher / student を兼ねる設計も多いです。たとえば teacher には参照解や追加コンテキスト、自然言語フィードバック、privileged information を見せる。student は通常どおり問題だけを見る。そして student が自分で生成した rollout の上で、teacher と student の分布差を学習する。これは圧縮というより、「言語空間の中で自分に濃いフィードバックを返しながら学ぶ」手法と見たほうがしっくりきます。

## なぜ on-policy が効くのか

この流れで本質的なのは self よりも on-policy だと感じています。固定データに対する off-policy 学習では、モデルが実際にどんな思考過程や言い回しを出力するか、その分布変化を取り込みにくい。一方 on-policy なら、いまのモデルが実際に生成した prefix の上で修正がかけられます。

これは imitation learning の文脈では昔から重要だった話です。特に、推論モデルでは、最終回答だけではなく途中の言語的な推論列そのものが性能を作っています。だから、各トークン位置で「この状況なら次にどういう分布を出すべきか」を教えられることに価値があります。[On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) が良い学習ができるのはこの点が大きいと思われます。RL のように最後にスカラー報酬を返すのではなく、教師側が token-level に密な信号を返しやすいので、学習の粒度を細かくできます。

## 適用範囲が広い

このパラダイムが面白いのは、reasoning 強化だけに閉じないことです。まず分かりやすいのは、参照解がある推論タスクです。teacher に正解や golden trajectory を見せて student を改善する、というのは [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) の中心的なアイデアです。

次に、継続学習や知識獲得でも有効そうです。[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897) では、新しいデータを学習しながら既存能力や reasoning style を壊しにくいことが報告されています。これは実務的に重要で、低品質な SFT データや answer-only データをそのまま食わせるとモデルを壊しやすい場面でも、on-policy な自己蒸留なら緩和できる可能性があります。加えて、この論文の Knowledge Acquisition 実験を見ると、OPSD系は単に新しい振る舞いを積み増すだけでなく、注入した知識をある程度モデル内部に取り込み、OODな質問にも効く形で知識獲得に使えるかもしれません。

![継続学習での性能トレードオフ（Self-Distillation Enables Continual Learning, Figure 4）](../assets/sdcl-figure4-continual-learning-tradeoff.png)

さらに、自然言語フィードバックとの相性も良いです。「ここが違う」「もっと簡潔に」「この前提を明示して」といったテキストフィードバックは、0/1 の reward よりもはるかに密です。人間のフィードバックも本来はこちらに近いので、binary reward 中心の RL から rich verbal feedback へ広がっていく流れの受け皿としても OPSD は筋が良いと思います。

加えて、teacher と student のあいだで tokenizer や logit 空間がそのまま揃わないケースも論点になりやすいです。実際の蒸留では、同じ語彙上で素直に full-vocabulary distillation できるとは限りません。その制約の中で、補助的なコンテキストや privileged information をどう使って teacher 信号を作るか、という問題設定は今後増えそうです。

## 現状の制約とこれからの論点

とはいえ、この流れはまだ始まったばかりで、未整理の論点も多いです。

ひとつは teacher 品質の問題です。OPSD は teacher を自分の中に作れるのが強みですが、逆に言えば teacher の質はモデル自身の能力に依存します。小さいモデルで効きにくい可能性や、どの程度のベース能力があれば self teacher が成立するのかは、重要です。

もうひとつは、teacher が答えや参照解を知った状態で出す logit を、そのまま student に学ばせてよいのかという問題です。特に推論タスクでは、teacher は privileged information を見たうえで次トークン分布を出しているので、その分布が「答えを知らない student が本来持つべき分布」と素直に一致するとは限りません。ここは OPSD の本質的な論点で、どの情報を teacher に持たせると有効で、どこから leakage や過剰誘導になるのかは、まだ研究余地があると思います。

現時点で何か1つの完成形が見えているというより、「SFTとRLのあいだに面白い設計空間が見つかり、その中で有望な手法群が出始めた」と捉えるのが自然です。個人的には、ここから色々掘れるテーマなのではないかと考えています。

## OPSD は新しい学習パラダイムになりそう

2026年初頭の論文群をまとめて見ると、OPSD はもう一時的なバズというより、かなり独立した設計空間として見えてきています。SFT が「データをなぞって学ぶ」、RL が「報酬で探索しながら学ぶ」だとすると、OPSD は「追加のコンテキストや言語的な情報で教師信号を密にし、それを使って学ぶ」パラダイムとして整理できそうです。

もっと大きく見ると、LLM が「言語空間で学習し、外挿し、自己修正する」方向に進み始めている、とも言えます。reasoning も prompt optimization も、その場で言語的に探索して性能を引き上げる手法です。OPSD ムーブメントは、そのポストトレーニング版として本質的だと思います。

特にモデル開発の観点で大きいのは、モデルを大きく壊さずに、追加の情報を使って安定して学習を進められる可能性があることです。高品質な教師データをそのままなぞるだけでもなく、最終報酬だけで強く押すだけでもなく、言語的な情報を中間に挟みながら少しずつ振る舞いを修正していける。この感覚は、reasoning の改善だけでなく、知識獲得や継続学習、モデルの振る舞い調整まで含めて重要だと思っています。


## 参考

- Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models: https://arxiv.org/abs/2601.18734
- On-Policy Distillation (Thinking Machines Lab): https://thinkingmachines.ai/blog/on-policy-distillation/
- Reinforcement Learning via Self-Distillation: https://arxiv.org/abs/2601.20802
- Self-Distillation Enables Continual Learning: https://arxiv.org/abs/2601.19897
- Expanding the Capabilities of Reinforcement Learning via Text Feedback: https://arxiv.org/abs/2602.02482
- Privileged Information Distillation for Language Models (π-Distill): https://arxiv.org/abs/2602.04942


# title options
- LLMの新たな学習パラダイムとしてのSelf-Distillation
- なぜ今On-Policy Self-Distillationなのか
- SFTでもRLでもない: self-distillationムーブメントを整理する
- 2026年初頭のself-distillation論文群から見る新しい潮流
- LLMは言語空間でどう学び始めたのか: self-distillationの台頭
