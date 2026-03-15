以下、相手にそのまま渡せる文面です。

---

## レビューコメント

全体の方向性はよいですが、現状は API の一貫性・抽象化の粒度・ND 周りの表現・推論パスの型設計にまだ揺れがあります。`ggml-rs` 本体は `Context` / `Tensor<'ctx>` を中心にした安全ラッパで、`lib.rs` でも “focused subset of ggml” と明示されています。この設計方針自体は維持しつつ、以下は**すべて対応してください**。 ([GitHub][1])

### 1. `with_context` / `with_no_alloc_context` の scoped helper を追加してください

`Context::new_bytes` / `Context::new_no_alloc_bytes` を毎回呼び出させるのではなく、クロージャベースの scoped helper を追加してください。これは中核 API の置き換えではなく、**現行の lifetimed な設計を保ったまま入口を ergonomic にする補助レイヤ**として入れるべきです。`for<'ctx>` を使い、`Tensor<'ctx>` がスコープ外へ逃げない形で定義してください。現状の `Context` / `Tensor<'ctx>` 中心の設計と矛盾しません。 ([GitHub][1])

```rust
pub fn with_context<R>(
    mem: Bytes,
    f: impl for<'ctx> FnOnce(&'ctx Context) -> Result<R>,
) -> Result<R> {
    let ctx = Context::new_bytes(mem)?;
    f(&ctx)
}

pub fn with_no_alloc_context<R>(
    mem: Bytes,
    f: impl for<'ctx> FnOnce(&'ctx Context) -> Result<R>,
) -> Result<R> {
    let ctx = Context::new_no_alloc_bytes(mem)?;
    f(&ctx)
}
```

### 2. テスト面積を API 面積に見合うところまで広げてください

公開 API に対してテストがまだ薄いです。`tests/` にあるのは `ggml_simple_ctx.rs`, `ggml_test_cont.rs`, `ggml_upstream_suite.rs`, `gguf_roundtrip.rs` の 4 本ですが、`compute.rs` には `new_tensor_3d/4d`, `repeat`, `concat`, backend allocation / read / write、さらに各種 tensor operation が並んでいます。ここは**すべてカバーを増やしてください**。特に backend path、ND tensor path、reshape/view/permute 系、エラー系、境界値を追加してください。 ([GitHub][2])

### 3. `inference.rs` の `NonZeroUsize` newtype 群は macro で共通化してください

`InFeatures`, `OutFeatures`, `HiddenFeatures`, `FfnFeatures` は、いずれも `NonZeroUsize` を包んで `new` / `get` を持つ同型の実装です。ここは**ジェネリクスではなく macro で共通化してください**。公開 API 上はそれぞれ別型のまま残し、型安全性とエラーメッセージの文脈は維持したまま、実装重複だけを消してください。`FeatureCount<T>` にまとめるより、各型を macro で生成する方がこのコードベースには適しています。 ([GitHub][3])

### 4. `&[f32]` を受ける公開 API は `impl AsRef<[T]>` に変え、要素型 `T` まで generic にしてください

`linear_inference_with_weights_repeats` をはじめ、推論系 API は現在 `input: &[f32]` を受け、内部でも `new_f32_tensor_2d_shape`、`write_data_backend(input)`, `read_data_backend::<f32>()` という形で `f32` に固定されています。これはコンテナ型だけでなく、**要素型そのものも generic にするべきです**。 `AsRef<[f32]>` に留めず、`LinearWeights<T>`, `MlpWeights<T>`, `input: impl AsRef<[T]>`, backend read/write, decode, report の output まで含めて `T` に統一してください。 ([GitHub][3])

具体的には、少なくとも次を揃えてください。

* `LinearWeights` / `MlpWeights` / `AttentionWeights` の内部 `Vec<f32>` を `Vec<T>` に変更
* `values(&self) -> &[f32]` を `values(&self) -> &[T]` に変更
* `input: &[f32]` を `input: impl AsRef<[T]>` に変更
* `LinearInferenceReport` / `MlpInferenceReport` / attention 系 report の `output: Vec<f32>` を `Vec<T>` に変更
* `tensor_f32_values` / `decode_tensor_f32_into` 相当を generic decode API に置き換え
* `Context::new_f32_tensor_*` 依存を外し、`Type` と `T` から tensor 型を解決するように変更
* `read_data_backend::<f32>()` / `write_data_backend` の経路を `T` と整合するように整理

要するに、推論パス全体から `f32` 固定を排除し、**入力・重み・出力・decode・backend I/O を全部ひとつの型パラメータで貫いてください**。 ([GitHub][3])

### 5. `model.rs` の metadata access は trait ベースの generic API に統合してください

`GgufModel` には `kv_value`, `kv_string`, `kv_usize`, `kv_f32` があり、数値変換ロジックが分散しています。ここは `TryFromGgufValue` のような trait を定義して、**metadata access を generic API に統合してください**。 `kv_value_as<T>()` を追加し、`usize`, `f32` などの所有値系はそこへ寄せてください。現状の個別メソッドは整理・削除してかまいません。 ([GitHub][4])

### 6. 文字列引数は `impl AsRef<str>` に統一してください

`find_tensor`, `kv_entry`, `kv_value`, `kv_string`, `tensor_f32_values`, `decode_tensor_f32_into` など、名前やキーを受ける API は現在 `&str` 固定です。ここは**全部 `impl AsRef<str>` に変えてください**。呼び出し側が `String` / `Cow<str>` をそのまま渡せるようにし、lookup 系 API の ergonomics を揃えてください。 ([GitHub][4])

### 7. GGUF decode API も `f32` 固定をやめて generic にしてください

現状の model/inference 経路では `tensor_f32_values`, `decode_tensor_f32_into`, `decode_tensor_data_to_f32` が中心になっており、推論ヘルパもそれに依存しています。ここは `decode_tensor_data_to::<T>()` 相当の generic API を導入し、`GgufModel` 側も `tensor_values<T>()` / `decode_tensor_into<T>()` の形に変更してください。`f32` 専用 helper は互換レイヤではなく削ってかまいません。推論系 generic 化とセットで必ずやってください。 ([GitHub][5])

### 8. ND tensor API は alias 追加だけで終わらせず、generic 化まで明示的にやってください

`compute.rs` には `new_tensor_3d` / `new_tensor_4d` があり、`shape.rs` には `Cols`, `Rows`, `Shape2D`, `StaticShape2D`, `Shape2DSpec` など 2D 中心の表現があります。ここは alias を足す程度で止めず、**ND shape 表現・introspection・tensor constructor をまとめて generic 化してください**。 ([GitHub][5])

最低限、以下は全部やってください。

* `Shape3D`, `Shape4D` を追加
* `Dims<const N: usize>` かそれに相当する汎用 shape 表現を追加
* `rank()` / `dims()` / `shape_nd()` 相当の introspection を追加
* `new_tensor<const N: usize>(ty, dims)` を導入し、1D〜4D を統一
* typed tensor 側も 2D 専用に閉じず、必要な範囲で ND 化
* 既存の `new_tensor_1d_len`, `new_tensor_2d_shape`, `new_tensor_3d`, `new_tensor_4d` は薄い委譲に落とす

ここは「必要なら generic 化」ではなく、**明示的に generic 化までやってください**。alias 追加だけでは不十分です。 ([GitHub][5])

### 9. `shape.rs` の semantic wrapper 群も整理してください

`Cols`, `Rows`, `Length`, `TensorIndex`, `ThreadCount`, `Bytes` はいずれも同じ構造の薄い wrapper です。ここも個別実装を繰り返さず、macro で定義を生成してください。2D shape だけでなく、こうした semantic quantity 群も一貫した書き方に揃えてください。 ([GitHub][6])

### 10. `Type` は推論系 generic 化と整合するところまで拡張してください

現状の safe wrapper で公開されている `Type` は `F32` と `I32` だけです。推論パスを generic にするなら、`Type` 側もそれと整合するように拡張してください。少なくとも `T` と `Type` を対応付ける trait を導入し、tensor 生成・decode・backend I/O がすべて同じ型情報を使うようにしてください。 `f32` 固定の constructor や decode helper を残したままにせず、型レベルで一本化してください。 ([GitHub][7])

### 11. backend path の API とサンプルを増やしてください

`allocate_tensors`, `write_data_backend`, `read_data_backend` はすでにありますが、利用例・検証・テストが不足しています。backend path は safe wrapper の大事な価値なので、**サンプルとテストを増やして、CPU / Metal を含めた典型パスを明示してください**。 `with_context` 系 helper を入れるなら、それを使った backend compute のサンプルも追加してください。 ([GitHub][5])

### 12. README / 利用導線もコード変更に合わせて整理してください

`lib.rs` には `link-system` と `GGML_RS_LIB_DIR`, `GGML_RS_LIB_DIRS`, `GGML_RS_LIBS` の説明がありますが、導線はまだ薄いです。API をここまで整理するなら、README も必ず更新し、セットアップ、backend 利用、GGUF decode、generic tensor API、scoped helper の使い方まで含めて一貫した導線にしてください。 ([GitHub][1])

## 実施方針

この変更は部分最適で止めず、**全部やる前提でまとめて進めてください**。
具体的には、次の方針を最後まで貫いてください。

1. `Context` / `Tensor<'ctx>` の中核設計は維持する
2. その上で scoped helper を追加する
3. newtype / wrapper の重複は macro で消す
4. 文字列入力は `AsRef<str>` に統一する
5. 推論パスは container だけでなく要素型 `T` まで generic にする
6. GGUF decode も generic にする
7. ND shape / constructor / introspection は alias 追加で終わらせず generic 化までやる
8. backend path とテスト、README まで含めて一式そろえる

中途半端な互換レイヤや暫定実装ではなく、**最終形に寄せる方向で全部整理してください**。

[1]: https://raw.githubusercontent.com/Sanzentyo/ggml-rs/master/src/lib.rs "raw.githubusercontent.com"
[2]: https://github.com/Sanzentyo/ggml-rs/tree/master/tests "ggml-rs/tests at master · Sanzentyo/ggml-rs · GitHub"
[3]: https://raw.githubusercontent.com/Sanzentyo/ggml-rs/master/llama-rs/src/inference.rs "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Sanzentyo/ggml-rs/master/llama-rs/src/model.rs "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Sanzentyo/ggml-rs/master/src/compute.rs "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Sanzentyo/ggml-rs/master/src/shape.rs "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/Sanzentyo/ggml-rs/master/src/types.rs "raw.githubusercontent.com"
