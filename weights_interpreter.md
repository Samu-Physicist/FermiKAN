# 取り出した重みの翻訳について

## 1. FermiKANの数式
数学的にはNBO軌道を学習している？
- 1電子軌道

$$
\phi_k = \sum_{I \in \text{Atoms}} \sum_{(n, l) \in \text{Pool}} C_{k, I, n, l} \cdot \left[ R_{n, l}(|\mathbf{r}_{iI}|_{\text{shifted}}, \xi(\mathbf{h})) \right] \cdot \left( |\mathbf{r}_{iI}|_{\text{true}}^l \cdot \mathcal{P}_l(\mathbf{q}_{4D}) \right) \cdot \exp(\text{Envelope}(|\mathbf{r}_{iI}|_{\text{true}}))
$$

- scalar backflow

$$ \mathbf{q}_i = \mathbf{r}_i \cdot \exp\left( \sum_{j \neq i} \eta(r_{ij}^2) \right), \quad \eta(r_{ij}^2) = \text{KAN}_{backflow}(r_{ij}) \cdot \exp(-\alpha r_{ij}^2) $$

- envelope

$$
\text{Envelope}(r, \mathbf{h}) = -\xi(\mathbf{h}) r - \frac{Z_I/n - \xi(\mathbf{h})}{\beta} (1 - e^{-\beta r})
$$

$$ \xi(\mathbf{h}) = \text{softplus}\Big( \xi_{\text{static}} + \Delta\xi_{\text{static}}(r) + \Delta\xi_{\text{dynamic}}(\mathbf{h}) \Big) $$

## 2. 重みの翻訳

厳密に原子の局在化や軌道の寄与率を決定するためには、$C$ だけでなく $\mathcal{P}$ の重み $W$ まで掛け合わせた『実効的な係数』を見る必要がある。

$$ \phi_k = \sum_{I} \sum_{n,l} C_{k, I, n, l} \cdot R_{n,l} \cdot \underbrace{\left( \sum_{m} W_{k, I, n, l, m} M_{l,m}(\mathbf{q}) \right)}_{\text{Angular KAN } \mathcal{P}_l} \cdot \text{Env} $$

どの原子（$I$）の、どの軌道（$s, p_x, d_{z^2}$ など）が、どのMO（$k$）に局在しているかを完全に解釈するためには、$C$ と $W$ を掛け合わせた実効的なLCAOテンソル $\tilde{C}$ を計算して評価する必要があります。

$$ \tilde{C}_{k, I, n, l, m} = C_{k, I, n, l} \times W_{k, I, n, l, m} $$

- 例えば「水素原子Aの1s軌道」の寄与が見たい場合は、$\tilde{C}_{k, A, 1, 0, 0}$ を見ます。

- 「結合軸（z軸）方向への $p_z$ 分極（混成軌道の形成）」を確認する場合は、$\tilde{C}_{k, A, 1, 1, z}$ （$n=1, l=1$ の $p_z$ 軌道の係数）がどれくらい大きく育っているかを見ることになります。水素原子でありながら自律的に $s-p$ 混成を起こしていることがここから読み取れます。

電子密度行列 $P$ は単なる行列の積で一発で求まります。

$$ P = \tilde{C} \cdot \tilde{C}^T $$

（※アップスピンとダウンスピンで別々の $C$ を持っている場合は、$P^\alpha = \tilde{C}^\alpha (\tilde{C}^\alpha)^T$、$P^\beta = \tilde{C}^\beta (\tilde{C}^\beta)^T$ として求めます）

重なり行列（Overlap Matrix）$S$ は、2つの基底関数（原子軌道など）$\chi_\mu$ と $\chi_\nu$ が空間的にどれくらい重なっているかを示す積分値の行列です。 $$ S_{\mu\nu} = \int \chi_\mu^*(\mathbf{r}) \chi_\nu(\mathbf{r}) d\mathbf{r} $$

1. 電子の数を正しく数えるとき（Mullikenポピュレーション解析など）
例えば、「原子Aに電子が何個いるか」を計算したいとします。 もし直交基底なら、単純に密度行列 $P$ の「原子Aの対角成分」を足し合わせれば終わりです。 しかし、実際の原子軌道は重なっているため、「原子Aと原子Bの中間（共有結合の部分）にいる電子」をどちらの原子にカウントするか決める必要があります。このとき、$P \times S$ という行列積を計算することで、重なり部分の電子をAとBに公平に分配することができます。

2. 対角化して「自然軌道」を取り出すとき（NBO解析など）
先ほど「$P$ をブロック対角化する」とお話ししましたが、実は非直交な基底のまま普通の対角化（固有値問題を解くこと）をやってしまうと、出てくる固有ベクトルが直交せず、固有値（電子の数）も合計が元の電子数と合わなくなってしまいます。 そのため、NBO解析などのプログラムの内部では、まず $S$ を使って基底関数をムリヤリ直交化（Löwdin直交化などと呼ばれます）してから、$P$ を対角化する という手順を踏んでいます。

### スピン対称性の破れ（Spin Contamination）の検証
$\tilde{C}$ からアップスピンの密度行列 $P^\alpha$ とダウンスピンの密度行列 $P^\beta$ を別々に計算し、両者を比較することで、ネットワークが「UHF的な対称性の破れ（スピン汚染）」を起こしているかどうかを検証できます。
もし $P^\alpha$ と $P^\beta$ で電子の空間分布が大きく異なっていれば（例: $\alpha$が左の原子、$\beta$が右の原子に偏る）、それは電子相関を表現するためにネットワークが意図的にスピン対称性を破った（ズルをした）証拠になります。逆に両者が幾何学的に一致していれば、正しい一重項（Singlet）の対称性を保ったまま Jastrow 因子等で正しく相関を取り込めていることの証明になります。