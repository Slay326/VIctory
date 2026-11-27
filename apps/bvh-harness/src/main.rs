//! `bvh-harness <capture folder> [--runs N] [--all] [--barriers] [--card]
//! [--warp] [--dump <folder>]`: runs a BVH builder capture
//! (`VICTORY_BVH_CAPTURE`) alone and checks it (`victory_gpu::bvh_replay`).
//! Exit code 0: the game's result reproduced; 1: not; 2: cannot replay.

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
