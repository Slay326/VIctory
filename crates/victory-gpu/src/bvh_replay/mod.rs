//! Running BVH builder submissions again on their own (local test branch):
//! [`format`] of a capture, `capture` (game side, diagnostics builds) and
//! [`harness`] (a D3D12 device of its own). Captures hold game data and are
//! never committed.

pub mod format;

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod capture;
#[cfg(windows)]
pub mod harness;

/// The adapter behind `device` and its driver version, for the capture's
/// and the harness's reports.
#[cfg(windows)]
pub(crate) fn adapter_text(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> String {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4};
    use windows::core::Interface;
    // SAFETY: plain DXGI queries.
    unsafe {
        let luid = device.GetAdapterLuid();
        let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
            return "?".into();
        };
        let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter1>(luid) else {
            return "?".into();
        };
        let Ok(desc) = adapter.GetDesc1() else {
            return "?".into();
        };
        let name = String::from_utf16_lossy(&desc.Description)
            .trim_end_matches('\0')
            .to_string();
        let driver = adapter
            .CheckInterfaceSupport(&windows::Win32::Graphics::Dxgi::IDXGIDevice::IID)
            .map(|v| {
                let v = v as u64;
                format!(
                    "{}.{}.{}.{}",
                    v >> 48,
                    (v >> 32) & 0xffff,
                    (v >> 16) & 0xffff,
                    v & 0xffff
                )
            })
            .unwrap_or_else(|_| "?".into());
        format!("{name} driver {driver}")
    }
}
