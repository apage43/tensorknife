# tensorknife

convert named ndarrays with metadata between formats

* currently just converts pytorch saved weights to safetensors, but without needing pytorch or python:

```
cargo build --release
./target/release/tensorknife ~/path/to/model-*.bin output-file.safetensors
```

or

```
cargo run --release -- ~/path/to/model-*.bin output-file.safetensors
```

converts split mpt-7b pytorch files into a single safetensor file in 10 seconds on a M2 mac:

```
❯ time ./target/release/tensorknife ~/model/mpt-7b-chat/pytorch_model*.bin mpt-7b-chat.safetensors
./target/release/tensorknife ~/model/mpt-7b-chat/pytorch_model*.bin   3.11s user 4.40s system 71% cpu 10.430 total
❯ du -sh ~/model/mpt-7b-chat/pytorch_model*.bin *.safetensors                                
9.3G	/Users/aaron/model/mpt-7b-chat/pytorch_model-00001-of-00002.bin
3.1G	/Users/aaron/model/mpt-7b-chat/pytorch_model-00002-of-00002.bin
 12G	mpt-7b-chat.safetensors
```
