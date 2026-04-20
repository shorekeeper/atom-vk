// Minimal raw Win32 API bindings. We avoid pulling in winit, windows-rs, or
// windows-sys; only the system import libraries (user32, kernel32) are linked.

#![allow(non_snake_case, non_camel_case_types, dead_code)]

use std::ffi::c_void;

pub type HANDLE    = *mut c_void;
pub type HINSTANCE = HANDLE;
pub type HWND      = HANDLE;
pub type HMODULE   = HANDLE;
pub type HICON     = HANDLE;
pub type HCURSOR   = HANDLE;
pub type HBRUSH    = HANDLE;
pub type HMENU     = HANDLE;
pub type LRESULT   = isize;
pub type WPARAM    = usize;
pub type LPARAM    = isize;
pub type ATOM      = u16;
pub type DWORD     = u32;
pub type UINT      = u32;
pub type BOOL      = i32;
pub type LONG      = i32;

pub type WNDPROC = unsafe extern "system" fn(HWND, UINT, WPARAM, LPARAM) -> LRESULT;

#[repr(C)]
pub struct WNDCLASSEXW {
    pub cbSize:        UINT,
    pub style:         UINT,
    pub lpfnWndProc:   WNDPROC,
    pub cbClsExtra:    i32,
    pub cbWndExtra:    i32,
    pub hInstance:     HINSTANCE,
    pub hIcon:         HICON,
    pub hCursor:       HCURSOR,
    pub hbrBackground: HBRUSH,
    pub lpszMenuName:  *const u16,
    pub lpszClassName: *const u16,
    pub hIconSm:       HICON,
}

#[repr(C)]
pub struct POINT { pub x: LONG, pub y: LONG }

#[repr(C)]
pub struct MSG {
    pub hwnd:     HWND,
    pub message:  UINT,
    pub wParam:   WPARAM,
    pub lParam:   LPARAM,
    pub time:     DWORD,
    pub pt:       POINT,
    pub lPrivate: DWORD,
}

#[repr(C)]
pub struct RECT { pub left: LONG, pub top: LONG, pub right: LONG, pub bottom: LONG }

// Window styles
pub const WS_OVERLAPPEDWINDOW: DWORD = 0x00CF0000;
pub const WS_VISIBLE:          DWORD = 0x10000000;

// Window messages
pub const WM_DESTROY: UINT = 0x0002;
pub const WM_CLOSE:   UINT = 0x0010;
pub const WM_QUIT:    UINT = 0x0012;
pub const WM_KEYDOWN: UINT = 0x0100;
pub const WM_SIZE:    UINT = 0x0005;

pub const WM_MOUSEMOVE:   UINT = 0x0200;
pub const WM_LBUTTONDOWN: UINT = 0x0201;
pub const WM_LBUTTONUP:   UINT = 0x0202;
pub const WM_RBUTTONDOWN: UINT = 0x0204;
pub const WM_RBUTTONUP:   UINT = 0x0205;
pub const WM_MOUSEWHEEL:  UINT = 0x020A;

// PeekMessage flags
pub const PM_REMOVE: UINT = 0x0001;

// ShowWindow commands
pub const SW_SHOW: i32 = 5;

// CreateWindow defaults
pub const CW_USEDEFAULT: i32 = 0x80000000u32 as i32;

// Virtual key codes
pub const VK_ESCAPE: WPARAM = 0x1B;
pub const VK_LEFT:   WPARAM = 0x25;
pub const VK_RIGHT:  WPARAM = 0x27;
pub const VK_UP:     WPARAM = 0x26;
pub const VK_DOWN:   WPARAM = 0x28;
pub const VK_PRIOR:  WPARAM = 0x21; // PageUp
pub const VK_NEXT:   WPARAM = 0x22; // PageDown

pub static mut MOUSE_X: i32 = 0;
pub static mut MOUSE_Y: i32 = 0;
pub static mut MOUSE_DX: i32 = 0;
pub static mut MOUSE_DY: i32 = 0;
pub static mut MOUSE_LEFT: bool = false;
pub static mut MOUSE_RIGHT: bool = false;
pub static mut WHEEL_DELTA_ACCUM: f32 = 0.0;
static mut LAST_MOUSE_X: i32 = 0;
static mut LAST_MOUSE_Y: i32 = 0;
static mut MOUSE_INITIALIZED: bool = false;


// Standard cursor identifier
pub fn IDC_ARROW() -> *const u16 { 32512usize as *const u16 }

#[link(name = "user32")]
unsafe extern "system" {
    pub fn RegisterClassExW(lpwcx: *const WNDCLASSEXW) -> ATOM;

    pub fn CreateWindowExW(
        dwExStyle: DWORD, lpClassName: *const u16, lpWindowName: *const u16,
        dwStyle: DWORD, X: i32, Y: i32, nWidth: i32, nHeight: i32,
        hWndParent: HWND, hMenu: HMENU, hInstance: HINSTANCE,
        lpParam: *mut c_void,
    ) -> HWND;

    pub fn ShowWindow(hWnd: HWND, nCmdShow: i32) -> BOOL;
    pub fn UpdateWindow(hWnd: HWND) -> BOOL;
    pub fn DefWindowProcW(hWnd: HWND, Msg: UINT, wParam: WPARAM, lParam: LPARAM) -> LRESULT;
    pub fn PostQuitMessage(nExitCode: i32);
    pub fn DestroyWindow(hWnd: HWND) -> BOOL;
    pub fn PeekMessageW(lpMsg: *mut MSG, hWnd: HWND,
                        wMsgFilterMin: UINT, wMsgFilterMax: UINT,
                        wRemoveMsg: UINT) -> BOOL;
    pub fn TranslateMessage(lpMsg: *const MSG) -> BOOL;
    pub fn DispatchMessageW(lpMsg: *const MSG) -> LRESULT;
    pub fn LoadCursorW(hInstance: HINSTANCE, lpCursorName: *const u16) -> HCURSOR;
    pub fn GetClientRect(hWnd: HWND, lpRect: *mut RECT) -> BOOL;
}

#[link(name = "kernel32")]
unsafe extern "system" {
    pub fn GetModuleHandleW(lpModuleName: *const u16) -> HMODULE;
}

// UTF-16 null-terminated literal helper.
pub fn wstr(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn loword_i16(v: u32) -> i16 { (v & 0xFFFF) as i16 }
fn hiword_i16(v: u32) -> i16 { ((v >> 16) & 0xFFFF) as i16 }

// Global flag toggled by the window procedure to signal a quit request.
// Written by WindowProc, read from main thread; access is serialized by
// the single-threaded message pump.
pub static mut QUIT_REQUESTED: bool = false;

// Global slots for keyboard input. The graphics layer polls them between
// frames. Same single-thread-message-pump invariant as above.
pub static mut KEY_PRESSED: [bool; 256] = [false; 256];
pub static mut KEY_CONSUMED: [bool; 256] = [true; 256];

pub unsafe extern "system" fn window_proc(
    hwnd: HWND, msg: UINT, w: WPARAM, l: LPARAM,
) -> LRESULT {
    match msg {
        WM_CLOSE => {
            QUIT_REQUESTED = true;
            0
        }
        WM_DESTROY => {
            PostQuitMessage(0);
            0
        }
        WM_KEYDOWN => {
            if w == VK_ESCAPE {
                QUIT_REQUESTED = true;
            }
            if w < 256 {
                KEY_PRESSED[w] = true;
                KEY_CONSUMED[w] = false;
            }
            0
        }
        WM_MOUSEMOVE => {
            let lp = l as u32;
            let x = loword_i16(lp) as i32;
            let y = hiword_i16(lp) as i32;
            if !MOUSE_INITIALIZED {
                LAST_MOUSE_X = x;
                LAST_MOUSE_Y = y;
                MOUSE_INITIALIZED = true;
            }
            MOUSE_DX += x - LAST_MOUSE_X;
            MOUSE_DY += y - LAST_MOUSE_Y;
            LAST_MOUSE_X = x;
            LAST_MOUSE_Y = y;
            MOUSE_X = x;
            MOUSE_Y = y;
            0
        }
        WM_LBUTTONDOWN => { MOUSE_LEFT  = true;  0 }
        WM_LBUTTONUP   => { MOUSE_LEFT  = false; 0 }
        WM_RBUTTONDOWN => { MOUSE_RIGHT = true;  0 }
        WM_RBUTTONUP   => { MOUSE_RIGHT = false; 0 }
        WM_MOUSEWHEEL => {
            // High word of WPARAM is the signed wheel delta in WHEEL_DELTA
            // units (120 per notch).
            let wp = w as u32;
            let delta = hiword_i16(wp) as f32 / 120.0;
            WHEEL_DELTA_ACCUM += delta;
            0
        }
        _ => DefWindowProcW(hwnd, msg, w, l),
    }
}

pub struct Window {
    pub hwnd:      HWND,
    pub hinstance: HINSTANCE,
    pub width:     u32,
    pub height:    u32,
}

pub struct MouseState {
    pub dx: f32,
    pub dy: f32,
    pub wheel: f32,
    pub left: bool,
    pub right: bool,
}

impl Window {
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        unsafe {
            let hinstance = GetModuleHandleW(std::ptr::null());
            let class_name = wstr("QuantumAtomVizClass");
            let title_w    = wstr(title);

            let wc = WNDCLASSEXW {
                cbSize:        std::mem::size_of::<WNDCLASSEXW>() as UINT,
                style:         0x0003, // CS_HREDRAW | CS_VREDRAW
                lpfnWndProc:   window_proc,
                cbClsExtra:    0,
                cbWndExtra:    0,
                hInstance:     hinstance,
                hIcon:         std::ptr::null_mut(),
                hCursor:       LoadCursorW(std::ptr::null_mut(), IDC_ARROW()),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName:  std::ptr::null(),
                lpszClassName: class_name.as_ptr(),
                hIconSm:       std::ptr::null_mut(),
            };
            let atom = RegisterClassExW(&wc);
            assert!(atom != 0, "RegisterClassExW failed");

            let hwnd = CreateWindowExW(
                0,
                class_name.as_ptr(),
                title_w.as_ptr(),
                WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                CW_USEDEFAULT, CW_USEDEFAULT,
                width as i32, height as i32,
                std::ptr::null_mut(), std::ptr::null_mut(),
                hinstance, std::ptr::null_mut(),
            );
            assert!(!hwnd.is_null(), "CreateWindowExW failed");

            ShowWindow(hwnd, SW_SHOW);
            UpdateWindow(hwnd);

            Window { hwnd, hinstance, width, height }
        }
    }

    pub fn poll_mouse(&self) -> MouseState {
        unsafe {
            let s = MouseState {
                dx: MOUSE_DX as f32,
                dy: MOUSE_DY as f32,
                wheel: WHEEL_DELTA_ACCUM,
                left:  MOUSE_LEFT,
                right: MOUSE_RIGHT,
            };
            MOUSE_DX = 0;
            MOUSE_DY = 0;
            WHEEL_DELTA_ACCUM = 0.0;
            s
        }
    }

    // Pump all pending messages. Returns false when a quit was requested.
    pub fn pump(&self) -> bool {
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                if msg.message == WM_QUIT {
                    QUIT_REQUESTED = true;
                }
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            !QUIT_REQUESTED
        }
    }

    // Read and consume a key press. Returns true exactly once per WM_KEYDOWN.
    pub fn consume_key(&self, vk: WPARAM) -> bool {
        unsafe {
            if vk >= 256 { return false; }
            if KEY_PRESSED[vk] && !KEY_CONSUMED[vk] {
                KEY_CONSUMED[vk] = true;
                KEY_PRESSED[vk] = false;
                true
            } else {
                false
            }
        }
    }
}