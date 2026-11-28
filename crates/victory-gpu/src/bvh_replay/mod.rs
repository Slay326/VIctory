//! Running BVH builder submissions again on their own (local test branch):
//! [`format`] of a capture, `capture` (game side, diagnostics builds) and
//! [`harness`] (a D3D12 device of its own). Captures hold game data and are
//! never committed.

pub mod format;

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod capture;
#[cfg(windows)]
pub mod harness;
pub mod reader;
pub mod reference;

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

/// Builder submissions ahead of the last one submitted (`done`) whose lists
/// the capture cuts around the builder (`capture`): only those due to be
/// captured, `at`, and only when one of them is the next or the one after
/// (a submission may be recorded while the one before is in flight). The
/// rest of the run keeps the lists as without capturing (Fable design point
/// 6: the capture must not change the regime).
pub fn cut_window(done: u64, at: &[u64]) -> bool {
    at.iter().any(|&n| n > done && n <= done + 2)
}

#[cfg(test)]
mod tests {
    use super::cut_window;

    #[test]
    fn lists_are_cut_only_next_to_a_captured_submission() {
        let at = [5, 6, 30, 45];
        // Up to the first capture: only the two before it.
        assert!(!cut_window(0, &at));
        assert!(!cut_window(2, &at));
        assert!(cut_window(3, &at));
        assert!(cut_window(4, &at));
        assert!(cut_window(5, &at));
        // Between captures the run is as without capturing.
        assert!(!cut_window(6, &at));
        assert!(!cut_window(20, &at));
        assert!(cut_window(28, &at));
        assert!(!cut_window(30, &at));
        // After the last one: never again.
        assert!(!cut_window(45, &at));
        assert!(!cut_window(1000, &at));
        assert!(!cut_window(0, &[]));
    }
}
