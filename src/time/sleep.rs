use std::thread;
use std::time;


/// Puts the current thread to sleep for at least the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality. It will never sleep less.
///
/// This function is blocking, and should not be used in `async` functions.
///
/// # Platform-specific behavior
///
/// On Unix platforms, the underlying syscall may be interrupted by a
/// spurious wakeup or signal handler. To ensure the sleep occurs for at least
/// the specified duration, this function may invoke that system call multiple
/// times.
/// Platforms which do not support nanosecond precision for sleeping will
/// have `dur` rounded up to the nearest granularity of time they can sleep for.
///
/// Currently, specifying a zero duration on Unix platforms returns immediately
/// without invoking the underlying [`nanosleep`] syscall, whereas on Windows
/// platforms the underlying [`Sleep`] syscall is always invoked.
/// If the intention is to yield the current time-slice you may want to use
/// [`yield_now`] instead.
///
/// [`nanosleep`]: https://linux.die.net/man/2/nanosleep
/// [`Sleep`]: https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-sleep
///
/// # Examples
///
/// ```no_run
/// use std::time;
/// use jkcutils_rs::time::sleep;
///
/// let now = time::Instant::now();
/// 
/// sleep(1);
/// 
/// assert!(now.elapsed() >= time::Duration::from_secs(1));
/// 
/// ```
#[inline]
pub fn sleep(secs: u64) {
    thread::sleep(time::Duration::from_secs(secs))
}


#[inline]
pub fn sleep_s(secs: u64) {
    thread::sleep(time::Duration::from_secs(secs))
}


#[inline]
pub fn sleep_ms(millis: u64) {
    thread::sleep(time::Duration::from_millis(millis))
}


#[inline]
pub fn sleep_us(micros: u64) {
    thread::sleep(time::Duration::from_micros(micros))
}


#[inline]
pub fn sleep_ns(nanos: u64) {
    thread::sleep(time::Duration::from_nanos(nanos))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sleep() {
        let now = time::Instant::now();
        sleep(1);
        assert!(now.elapsed() >= time::Duration::from_secs(1));
    }
}