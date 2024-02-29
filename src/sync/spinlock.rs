use std::fmt;
use std::ops::{Deref, DerefMut};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::sync::{TryLockResult, TryLockError, LockResult};
use super::poision;


/// A mutual exclusion primitive useful for protecting shared data
///
/// This spinlock will block threads waiting for the lock to become available. The
/// spinlock can be created via a [`new`] constructor. Each spinlock has a type parameter
/// which represents the data that it is protecting. The data can only be accessed
/// through the RAII guards returned from [`lock`] and [`try_lock`], which
/// guarantees that the data is only ever accessed when the spinlock is locked.
///
/// # Poisoning
///
/// The spinlockes in this module implement a strategy called "poisoning" where a
/// spinlock is considered poisoned whenever a thread panics while holding the
/// spinlock. Once a spinlock is poisoned, all other threads are unable to access the
/// data by default as it is likely tainted (some invariant is not being
/// upheld).
///
/// For a spinlock, this means that the [`lock`] and [`try_lock`] methods return a
/// [`Result`] which indicates whether a spinlock has been poisoned or not. Most
/// usage of a spinlock will simply [`unwrap()`] these results, propagating panics
/// among threads to ensure that a possibly invalid invariant is not witnessed.
///
/// A poisoned spinlock, however, does not prevent all access to the underlying
/// data. The [`PoisonError`] type has an [`into_inner`] method which will return
/// the guard that would have otherwise been returned on a successful lock. This
/// allows access to the data, despite the lock being poisoned.
///
/// [`new`]: Self::new
/// [`lock`]: Self::lock
/// [`try_lock`]: Self::try_lock
/// [`unwrap()`]: Result::unwrap
/// [`PoisonError`]: super::PoisonError
/// [`into_inner`]: super::PoisonError::into_inner
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use std::sync::mpsc::channel;
/// use jkcutils_rs::sync::SpinLock;
///
/// const N: usize = 10;
///
/// // Spawn a few threads to increment a shared variable (non-atomically), and
/// // let the main thread know once all increments are done.
/// //
/// // Here we're using an Arc to share memory among threads, and the data inside
/// // the Arc is protected with a spinlock.
/// let data = Arc::new(SpinLock::new(0));
///
/// let (tx, rx) = channel();
/// for _ in 0..N {
///     let (data, tx) = (Arc::clone(&data), tx.clone());
///     thread::spawn(move || {
///         // The shared state can only be accessed once the lock is held.
///         // Our non-atomic increment is safe because we're the only thread
///         // which can access the shared state when the lock is held.
///         //
///         // We unwrap() the return value to assert that we are not expecting
///         // threads to ever fail while holding the lock.
///         let mut data = data.lock().unwrap();
///         *data += 1;
///         if *data == N {
///             tx.send(()).unwrap();
///         }
///         // the lock is unlocked here when `data` goes out of scope.
///     });
/// }
///
/// rx.recv().unwrap();
/// ```
///
/// To recover from a poisoned spinlock:
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use jkcutils_rs::sync::SpinLock;
///
/// let lock = Arc::new(SpinLock::new(0_u32));
/// let lock2 = Arc::clone(&lock);
///
/// let _ = thread::spawn(move || -> () {
///     // This thread will acquire the spinlock first, unwrapping the result of
///     // `lock` because the lock has not been poisoned.
///     let _guard = lock2.lock().unwrap();
///
///     // This panic while holding the lock (`_guard` is in scope) will poison
///     // the spinlock.
///     panic!();
/// }).join();
///
/// // The lock is poisoned by this point, but the returned result can be
/// // pattern matched on to return the underlying guard on both branches.
/// let mut guard = match lock.lock() {
///     Ok(guard) => guard,
///     Err(poisoned) => poisoned.into_inner(),
/// };
///
/// *guard += 1;
/// ```
///
/// To unlock a spinlock guard sooner than the end of the enclosing scope,
/// either create an inner scope or drop the guard manually.
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use jkcutils_rs::sync::SpinLock;
///
/// const N: usize = 3;
///
/// let data_spinlock = Arc::new(SpinLock::new(vec![1, 2, 3, 4]));
/// let res_spinlock = Arc::new(SpinLock::new(0));
///
/// let mut threads = Vec::with_capacity(N);
/// (0..N).for_each(|_| {
///     let data_spinlock_clone = Arc::clone(&data_spinlock);
///     let res_spinlock_clone = Arc::clone(&res_spinlock);
///
///     threads.push(thread::spawn(move || {
///         // Here we use a block to limit the lifetime of the lock guard.
///         let result = {
///             let mut data = data_spinlock_clone.lock().unwrap();
///             // This is the result of some important and long-ish work.
///             let result = data.iter().fold(0, |acc, x| acc + x * 2);
///             data.push(result);
///             result
///             // The spinlock guard gets dropped here, together with any other values
///             // created in the critical section.
///         };
///         // The guard created here is a temporary dropped at the end of the statement, i.e.
///         // the lock would not remain being held even if the thread did some additional work.
///         *res_spinlock_clone.lock().unwrap() += result;
///     }));
/// });
///
/// let mut data = data_spinlock.lock().unwrap();
/// // This is the result of some important and long-ish work.
/// let result = data.iter().fold(0, |acc, x| acc + x * 2);
/// data.push(result);
/// // We drop the `data` explicitly because it's not necessary anymore and the
/// // thread still has work to do. This allows other threads to start working on
/// // the data immediately, without waiting for the rest of the unrelated work
/// // to be done here.
/// //
/// // It's even more important here than in the threads because we `.join` the
/// // threads after that. If we had not dropped the spinlock guard, a thread could
/// // be waiting forever for it, causing a deadlock.
/// // As in the threads, a block could have been used instead of calling the
/// // `drop` function.
/// drop(data);
/// // Here the spinlock guard is not assigned to a variable and so, even if the
/// // scope does not end after this line, the spinlock is still released: there is
/// // no deadlock.
/// *res_spinlock.lock().unwrap() += result;
///
/// threads.into_iter().for_each(|thread| {
///     thread
///         .join()
///         .expect("The thread creating or execution failed !")
/// });
///
/// assert_eq!(*res_spinlock.lock().unwrap(), 800);
/// ```
pub struct SpinLock<T: ?Sized> {
    locked: AtomicBool,
    poison: poision::Flag,
    data: UnsafeCell<T>,
}


unsafe impl<T: ?Sized + Send> Sync for SpinLock<T> {}


/// An RAII implementation of a "scoped lock" of a spinlock. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the spinlock can be accessed through this guard via its
/// [`Deref`] and [`DerefMut`] implementations.
///
/// This structure is created by the [`lock`] and [`try_lock`] methods on
/// [`SpinLock`].
///
/// [`lock`]: SpinLock::lock
/// [`try_lock`]: SpinLock::try_lock
pub struct SpinLockGuard<'a, T: ?Sized + 'a> {
    lock: &'a SpinLock<T>,
    poison: poision::Guard,
}


// unsafe impl<T: ?Sized> Send for SpinLockGuard<'_, T> {}
// unsafe impl<T: ?Sized + Sync> Sync for SpinLockGuard<'_, T> {}


impl<T> SpinLock<T> {
    /// Creates a new spinlock in an unlocked state ready for use.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SpinLock;
    /// 
    /// let spinlock = SpinLock::new(0);
    /// ```
    #[inline]
    pub const fn new(t: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            poison: poision::Flag::new(),
            data: UnsafeCell::new(t),
        }
    }
}


impl<T: ?Sized> SpinLock<T> {
    /// Acquires a spinlock, blocking the current thread until it is able to to so.
    /// 
    /// This function will block the local thread until it is available to acquire
    /// the spinlock. Upon returning, the thread is the only thread with the lock
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the spinlock will be unlocked.
    /// 
    /// The exact behavior on locking a spinlock in the thread which already holds
    /// the lock is left unspecified. However, this function will not return on
    /// the second call (it might panic or deadlock, for example).
    /// 
    /// # Errors
    /// 
    /// If another user of this spinlock panicked while holding the spinlock, then
    /// this call will return an error once the spinlock is acquired.
    /// 
    /// # Panics
    /// 
    /// This function might panic when called if the lock is already held by
    /// the current thread.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::sync::Arc;
    /// use std::thread;
    /// use jkcutils_rs::sync::SpinLock;
    ///
    /// let spinlock = Arc::new(SpinLock::new(0));
    /// let c_spinlock = Arc::clone(&spinlock);
    ///
    /// thread::spawn(move || {
    ///     *c_spinlock.lock().unwrap() = 10;
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*spinlock.lock().unwrap(), 10);
    /// ```
    pub fn lock(&self) -> LockResult<SpinLockGuard<'_, T>> {
        // swap: Stores a value into the bool, returning the previous value.
        while self.locked.swap(true, Ordering::Acquire) {
            std::hint::spin_loop();
        }

        unsafe { SpinLockGuard::new(self) }
    }

    /// Attempts to acquire this lock.
    /// 
    /// If the lock could not be acquired at this time, then [`Err`] is returned.
    /// Otherwise, an RAII guard is returned. The lock will be unlocked when the
    /// guard is dropped.
    /// 
    /// This function does not block.
    /// 
    /// # Errors
    /// 
    /// If another user of this spinlock panicked while holding the spinlock, then
    /// this call will return the [`Poisoned`] error if the spinlock would
    /// otherwise be acquired.
    /// 
    /// If the spinlock could not be acquired because it is alreadly locked, then
    /// this call will return the [`WouldBlock`] error.
    /// 
    /// [`Poisoned`]: TryLockError::Poisoned
    /// [`WouldBlock`]: TryLockError::WouldBlock
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::sync::Arc;
    /// use std::thread;
    /// use jkcutils_rs::sync::SpinLock;
    ///
    /// let spinlock = Arc::new(SpinLock::new(0));
    /// let c_spinlock = Arc::clone(&spinlock);
    ///
    /// thread::spawn(move || {
    ///     let mut lock = c_spinlock.try_lock();
    ///     if let Ok(ref mut spinlock) = lock {
    ///         **spinlock = 10;
    ///     } else {
    ///         println!("try_lock failed");
    ///     }
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*spinlock.lock().unwrap(), 10);
    /// ```
    pub fn try_lock(&self) -> TryLockResult<SpinLockGuard<'_, T>> {
        if self.locked.swap(true, Ordering::Acquire) {
            return Err(TryLockError::WouldBlock);
        }

        unsafe { Ok(SpinLockGuard::new(self)?) }
    }

    /// Immediately drops the guard, and consequently unlocks the spinlock.
    /// 
    /// This function is equivalent to calling [`drop`] on the guard but is more self-documenting.
    /// Alternately, the guard will be automatically dropped when it goes out of scope.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SpinLock;
    /// 
    /// let spinlock = SpinLock::new(0);
    /// 
    /// let mut guard = spinlock.lock().unwrap();
    /// *guard += 20;
    /// SpinLock::unlock(guard);
    /// ```
    pub fn unlock(guard: SpinLockGuard<'_, T>) {
        drop(guard);
    }

    /// Determines whether the spinlock is poisoned.
    /// 
    /// If another thread is active, the spinlock can still become poisoned at any
    /// time. You should not trust a `false` value for program correctness
    /// without additional synchronization.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::thread;
    /// use std::sync::Arc;
    /// use jkcutils_rs::sync::SpinLock;
    /// 
    /// let spinlock = Arc::new(SpinLock::new(0));
    /// let c_spinlock = Arc::clone(&spinlock);
    ///
    /// let _ = thread::spawn(move || {
    ///     let _lock = c_spinlock.lock().unwrap();
    ///     panic!(); // the spinlock gets poisoned
    /// }).join();
    /// assert_eq!(spinlock.is_poisoned(), true);
    /// ```
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poison.get()
    }

    /// Clear the poisoned state from a spinlock.
    /// 
    /// If the spinlock is poisoned, it will remain poisoned until this function is called. This
    /// allows recovering from a poisoned state and marking that it has recovered. For example, if
    /// the value is overwritten by a known-good value, then the spinlock can be marked as
    /// un-poisoned. Or possibly, the value could be inspected to determine if it is in a
    /// consistent state, and if so the poison is removed.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::thread;
    /// use std::sync::Arc;
    /// use jkcutils_rs::sync::SpinLock;
    /// 
    /// let spinlock = Arc::new(SpinLock::new(0));
    /// let c_spinlock = Arc::clone(&spinlock);
    ///
    /// let _ = thread::spawn(move || {
    ///     let _lock = c_spinlock.lock().unwrap();
    ///     panic!(); // the spinlock gets poisoned
    /// }).join();
    ///
    /// assert_eq!(spinlock.is_poisoned(), true);
    /// let x = spinlock.lock().unwrap_or_else(|mut e| {
    ///     **e.get_mut() = 1;
    ///     spinlock.clear_poison();
    ///     e.into_inner()
    /// });
    /// assert_eq!(spinlock.is_poisoned(), false);
    /// assert_eq!(*x, 1);
    /// ```
    #[inline]
    pub fn clear_poison(&self) {
        self.poison.clear();
    }

    /// Consumes this spinlock, returning the underlying data.
    /// 
    /// # Errors
    /// 
    /// If another user of this spinlock panicked while holding the spinlock, then
    /// this call will return an error instead.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SpinLock;
    /// 
    /// let spinlock = SpinLock::new(0);
    /// assert_eq!(spinlock.into_inner().unwrap(), 0);
    /// ```
    pub fn into_inner(self) -> LockResult<T>
    where
        T: Sized,
    {
        let data = self.data.into_inner();

        poision::map_result(self.poison.borrow(), |()| data)
    }

    /// Returns a mutable reference to the underlying data.
    /// 
    /// Since this call borrows the `SpinLock` mutably, no actual locking needs to
    /// take place -- the mutable borrow statically guarantees no locks exist.
    /// 
    /// Errors
    /// 
    /// If another user of this spinlock panicked while holding the spinlock, then
    /// this call will return an error instead.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SpinLock;
    /// 
    /// let mut spinlock = SpinLock::new(0);
    /// *spinlock.get_mut().unwrap() = 10;
    /// assert_eq!(*spinlock.lock().unwrap(), 10);
    /// ```
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        let data = self.data.get_mut();

        poision::map_result(self.poison.borrow(), |()| data)
    }
}


impl<T> From<T> for SpinLock<T> {
    /// Crates a new spinlock in an unlocked state ready for use.
    /// This is equivalent to [`SpinLock::new`]
    fn from(t: T) -> Self {
        Self::new(t)
    }
} 


impl<T: ?Sized + Default> Default for SpinLock<T> {
    /// Creates a `SpinLock<T>`, with the `Default` value for T.
    fn default() -> SpinLock<T> {
        SpinLock::new(Default::default())
    }
}


impl<T: ?Sized + fmt::Debug> fmt::Debug for SpinLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("SpinLock");
        match self.try_lock() {
            Ok(guard) => {
                d.field("data", &&*guard);
            }
            Err(TryLockError::Poisoned(err)) => {
                d.field("data", &&**err.get_ref());
            }
            Err(TryLockError::WouldBlock) => {
                d.field("data", &format_args!("<locked>"));
            }
        }
        d.field("poisoned", &self.poison.get());
        d.finish_non_exhaustive()
    }
}


impl<'a, T: ?Sized> SpinLockGuard<'a, T> {
    unsafe fn new(lock: &'a SpinLock<T>) -> LockResult<SpinLockGuard<'a, T>> {
        poision::map_result(lock.poison.guard(), |guard| Self { lock, poison: guard })
    }
}


impl<T: ?Sized> Deref for SpinLockGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { & *self.lock.data.get() }
    }
}


impl<T: ?Sized> DerefMut for SpinLockGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}


impl<T: ?Sized> Drop for SpinLockGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.poison.done(&self.poison);
        self.lock.locked.store(false, Ordering::Release);
    }
}


impl<T: ?Sized + fmt::Debug> fmt::Debug for SpinLockGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}


impl<T: ?Sized + fmt::Display> fmt::Display for SpinLockGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}


#[cfg(test)]
mod tests {
    use std::thread;
    use std::sync::Arc;
    use super::*;

    #[test]
    fn test_spinlock() {
        let spin_lock = SpinLock::new(Vec::new());

        std::thread::scope(|s| {
            s.spawn(|| spin_lock.lock().unwrap().push(1));
            s.spawn(|| {
                let mut spin_lock_guard = spin_lock.lock().unwrap();
                spin_lock_guard.push(2);
                std::thread::sleep(std::time::Duration::from_secs(1));
                spin_lock_guard.push(3);
            });
        });

        let spin_lock_guard = spin_lock.lock().unwrap();
        assert!(spin_lock_guard.as_slice() == [1, 2, 3] || spin_lock_guard.as_slice() == [2, 3, 1]);
    }

    #[test]
    fn test_spinlock_poison() {
        let spinlock = Arc::new(SpinLock::new(0));
        let c_spinlock = spinlock.clone();

        let _ = thread::spawn(move || {
            let _lock = c_spinlock.lock().unwrap();
            panic!();
        }).join();

        assert_eq!(spinlock.is_poisoned(), true);
        assert_eq!(spinlock.lock().is_err(), true);

        let x = spinlock.lock().unwrap_or_else(|mut e| {
            **e.get_mut() = 1;
            spinlock.clear_poison();
            e.into_inner()
        });
        assert_eq!(spinlock.is_poisoned(), false);
        assert_eq!(*x, 1);
    }
}