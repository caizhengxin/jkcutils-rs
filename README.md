# jkcutils-rs

[![Crates.io](https://img.shields.io/crates/v/jkcutils-rs)](https://crates.io/crates/jkcutils-rs)
[![Crates.io](https://img.shields.io/crates/d/jkcutils-rs)](https://crates.io/crates/jkcutils-rs)
[![License](https://img.shields.io/crates/l/jkcutils-rs)](LICENSE)

## Features

- [x] [sync](./src/sync/)
  + [x] [seqlock](./src/sync/seqlock.rs)
  + [x] [spinlock](./src/sync/spinlock.rs)
- [x] [time](./src/time/)
  + [x] [sleep](./src/time/sleep.rs) 

## Usage

> Cargo.toml

```toml
[dependencies]
jkcutils-rs = { git = "https://github.com/caizhengxin/jkcutils-rs.git" }
```

or:

```toml
[dependencies]
jkcutils-rs = "0.1.0"
```
