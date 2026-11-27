//! `cargo run -p victory-gpu --example bvh_harness -- <capture folder> [options]`:
//! the BVH builder / RT reader harness (`victory_gpu::bvh_replay::harness`)
//! as an example of this crate (senior 115: an app crate would add a
//! Cargo.lock entry, which is part of SOURCE_HASH). Exit code 0: the game's
//! result reproduced; 1: not; 2: cannot replay.

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    #[cfg(windows)]
    {
        let code = victory_gpu::bvh_replay::harness::main(&args)?;
        std::process::exit(code);
    }
    #[cfg(not(windows))]
    {
        let _ = args;
        anyhow::bail!("the harness needs D3D12 (Windows)");
    }
}
