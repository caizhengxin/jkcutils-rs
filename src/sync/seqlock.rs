use std::fmt;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use super::{poision, LockResult, TryLockResult, TryLockError};


/// A mutual exclusion primitive useful for protecting shared data
///
/// This seqlock [`write`] will block threads, [`read`] data without lock. The
/// seqlock can be created via a [`new`] constructor. Each seqlock has a type parameter
/// which represents the data that it is protecting. The data can only be accessed
/// through the RAII guards returned from [`write`] and [`try_write`] and [`read`]
/// and [`try_read`], which guarantees that the data is only ever accessed when
/// the seqlock is locked.
///
/// # Poisoning
///
/// The seqlockes in this module implement a strategy called "poisoning" where a
/// seqlock is considered poisoned whenever a thread panics while holding the
/// seqlock. Once a seqlock is poisoned, all other threads are unable to access the
/// data by default as it is likely tainted (some invariant is not being
/// upheld).
///
/// For a seqlock, this means that the [`write`] and [`try_write`] methods return a
/// [`Result`] which indicates whether a seqlock has been poisoned or not. Most
/// usage of a seqlock will simply [`unwrap()`] these results, propagating panics
/// among threads to ensure that a possibly invalid invariant is not witnessed.
///
/// A poisoned seqlock, however, does not prevent all access to the underlying
/// data. The [`PoisonError`] type has an [`into_inner`] method which will return
/// the guard that would have otherwise been returned on a successful lock. This
/// allows access to the data, despite the lock being poisoned.
///
/// [`new`]: Self::new
/// [`write`]: Self::write
/// [`try_write`]: Self::try_write
/// [`read`]: Self::read
/// [`try_read`]: Self::try_read
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
/// use jkcutils_rs::sync::SeqLock;
///
/// const N: usize = 10;
///
/// // Spawn a few threads to increment a shared variable (non-atomically), and
/// // let the main thread know once all increments are done.
/// //
/// // Here we're using an Arc to share memory among threads, and the data inside
/// // the Arc is protected with a seqlock.
/// let data = Arc::new(SeqLock::new(0));
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
///         let mut data = data.write().unwrap();
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
/// To recover from a poisoned seqlock:
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use jkcutils_rs::sync::SeqLock;
///
/// let lock = Arc::new(SeqLock::new(0_u32));
/// let lock2 = Arc::clone(&lock);
///
/// let _ = thread::spawn(move || -> () {
///     // This thread will acquire the seqlock first, unwrapping the result of
///     // `lock` because the lock has not been poisoned.
///     let _guard = lock2.write().unwrap();
///
///     // This panic while holding the lock (`_guard` is in scope) will poison
///     // the seqlock.
///     panic!();
/// }).join();
///
/// // The lock is poisoned by this point, but the returned result can be
/// // pattern matched on to return the underlying guard on both branches.
/// let mut guard = match lock.write() {
///     Ok(guard) => guard,
///     Err(poisoned) => poisoned.into_inner(),
/// };
///
/// *guard += 1;
/// ```
///
/// To unlock a seqlock guard sooner than the end of the enclosing scope,
/// either create an inner scope or drop the guard manually.
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use jkcutils_rs::sync::SeqLock;
///
/// const N: usize = 3;
///
/// let data_seqlock = Arc::new(SeqLock::new(vec![1, 2, 3, 4]));
/// let res_seqlock = Arc::new(SeqLock::new(0));
///
/// let mut threads = Vec::with_capacity(N);
/// (0..N).for_each(|_| {
///     let data_seqlock_clone = Arc::clone(&data_seqlock);
///     let res_seqlock_clone = Arc::clone(&res_seqlock);
///
///     threads.push(thread::spawn(move || {
///         // Here we use a block to limit the lifetime of the lock guard.
///         let result = {
///             let mut data = data_seqlock_clone.write().unwrap();
///             // This is the result of some important and long-ish work.
///             let result = data.iter().fold(0, |acc, x| acc + x * 2);
///             data.push(result);
///             result
///             // The seqlock guard gets dropped here, together with any other values
///             // created in the critical section.
///         };
///         // The guard created here is a temporary dropped at the end of the statement, i.e.
///         // the lock would not remain being held even if the thread did some additional work.
///         *res_seqlock_clone.write().unwrap() += result;
///     }));
/// });
///
/// let mut data = data_seqlock.write().unwrap();
/// // This is the result of some important and long-ish work.
/// let result = data.iter().fold(0, |acc, x| acc + x * 2);
/// data.push(result);
/// // We drop the `data` explicitly because it's not necessary anymore and the
/// // thread still has work to do. This allows other threads to start working on
/// // the data immediately, without waiting for the rest of the unrelated work
/// // to be done here.
/// //
/// // It's even more important here than in the threads because we `.join` the
/// // threads after that. If we had not dropped the seqlock guard, a thread could
/// // be waiting forever for it, causing a deadlock.
/// // As in the threads, a block could have been used instead of calling the
/// // `drop` function.
/// drop(data);
/// // Here the seqlock guard is not assigned to a variable and so, even if the
/// // scope does not end after this line, the seqlock is still released: there is
/// // no deadlock.
/// *res_seqlock.write().unwrap() += result;
///
/// threads.into_iter().for_each(|thread| {
///     thread
///         .join()
///         .expect("The thread creating or execution failed !")
/// });
///
/// assert_eq!(*res_seqlock.read().unwrap(), 800);
/// ```
pub struct SeqLock<T: ?Sized> {
    sequence: AtomicUsize,
    poison: poision::Flag,
    data: UnsafeCell<T>,
}


unsafe impl<T: ?Sized + Send> Sync for SeqLock<T> {}


/// An RAII implementation of a "scoped lock" of a seqlock. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the seqlock can be accessed through this guard via its
/// [`Deref`] and [`DerefMut`] implementations.
///
/// This structure is created by the [`write`] and [`try_write`] methods on
/// [`SeqLock`].
///
/// [`write`]: SeqLock::write
/// [`try_write`]: SeqLock::try_write
pub struct SeqLockWriteGuard<'a, T: ?Sized + 'a> {
    lock: &'a SeqLock<T>,
    poison: poision::Guard,
}


/// An RAII implementation of a "scoped lock" of a seqlock. When this structure is
/// dropped (falls out of scope), the lock will be unlocked.
///
/// The data protected by the seqlock can be accessed through this guard via its
/// [`Deref`] implementations.
///
/// This structure is created by the [`read`] and [`try_read`] methods on
/// [`SeqLock`].
///
/// [`read`]: SeqLock::read
/// [`try_read`]: SeqLock::try_read
pub struct SeqLockReadGuard<'a, T: ?Sized + 'a> {
    lock: &'a SeqLock<T>,
    poison: poision::Guard,
}


impl<T> SeqLock<T> {
    /// Creates a new seqlock in an unlocked state ready for use.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SeqLock;
    /// 
    /// let seqlock = SeqLock::new(0);
    /// ```
    #[inline]
    pub const fn new(t: T) -> Self {
        Self {
            sequence: AtomicUsize::new(0),
            poison: poision::Flag::new(),
            data: UnsafeCell::new(t),
        }
    }
}


impl<T: ?Sized> SeqLock<T> {
    /// Acquires a seqlock, read data without lock.
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
    /// use jkcutils_rs::sync::SeqLock;
    ///
    /// let seqlock = Arc::new(SeqLock::new(0));
    /// let c_seqlock = Arc::clone(&seqlock);
    ///
    /// thread::spawn(move || {
    ///     *c_seqlock.write().unwrap() = 10;
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*seqlock.read().unwrap(), 10);
    /// ```
    pub fn read(&self) -> LockResult<SeqLockReadGuard<'_, T>> {
        unsafe { SeqLockReadGuard::new(self) }
    }

    /// Attempts to acquire this seqlock, read data without lock.
    pub fn try_read(&self) -> TryLockResult<SeqLockReadGuard<'_, T>> {
        let seq1 = self.read_seqcount_begin();

        if seq1 % 2 != 0 {
            return Err(TryLockError::WouldBlock);
        }

        unsafe { Ok(SeqLockReadGuard::new(self)?) }
    }

    /// Acquires a seqlock, blocking the current thread until it is able to to so.
    /// 
    /// This function will block the local thread until it is available to acquire
    /// the seqlock. Upon returning, the thread is the only thread with the lock
    /// held. An RAII guard is returned to allow scoped unlock of the lock. When
    /// the guard goes out of scope, the seqlock will be unlocked.
    /// 
    /// The exact behavior on locking a seqlock in the thread which already holds
    /// the lock is left unspecified. However, this function will not return on
    /// the second call (it might panic or deadlock, for example).
    /// 
    /// # Errors
    /// 
    /// If another user of this seqlock panicked while holding the seqlock, then
    /// this call will return an error once the seqlock is acquired.
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
    /// use jkcutils_rs::sync::SeqLock;
    ///
    /// let seqlock = Arc::new(SeqLock::new(0));
    /// let c_seqlock = Arc::clone(&seqlock);
    ///
    /// thread::spawn(move || {
    ///     *c_seqlock.write().unwrap() = 10;
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*seqlock.read().unwrap(), 10);
    /// ```
    pub fn write(&self) -> LockResult<SeqLockWriteGuard<'_, T>> {
        while self.read_seqcount_begin() % 2 != 0 {
            std::hint::spin_loop();
        }

        unsafe { SeqLockWriteGuard::new(self) }
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
    /// If another user of this seqlock panicked while holding the seqlock, then
    /// this call will return the [`Poisoned`] error if the seqlock would
    /// otherwise be acquired.
    /// 
    /// If the seqlock could not be acquired because it is alreadly locked, then
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
    /// use jkcutils_rs::sync::SeqLock;
    ///
    /// let seqlock = Arc::new(SeqLock::new(0));
    /// let c_seqlock = Arc::clone(&seqlock);
    ///
    /// thread::spawn(move || {
    ///     let mut lock = c_seqlock.try_write();
    ///     if let Ok(ref mut seqlock) = lock {
    ///         **seqlock = 10;
    ///     } else {
    ///         println!("try_lock failed");
    ///     }
    /// }).join().expect("thread::spawn failed");
    /// assert_eq!(*seqlock.read().unwrap(), 10);
    /// ```
    pub fn try_write(&self) -> TryLockResult<SeqLockWriteGuard<'_, T>> {
        if self.read_seqcount_begin() % 2 != 0 {
            return Err(TryLockError::WouldBlock);
        }

        unsafe { Ok(SeqLockWriteGuard::new(self)?) }
    }

    #[inline]
    fn read_seqcount_begin(&self) -> usize {
        self.sequence.load(Ordering::Acquire)
    }

    #[inline]
    fn write_seqcount_begin(&self) {
        self.sequence.fetch_add(1, Ordering::Release);
    }

    #[inline]
    fn write_seqcount_end(&self) {
        self.sequence.fetch_add(1, Ordering::Release);
    }

    /// Determines whether the seqlock is poisoned.
    /// 
    /// If another thread is active, the seqlock can still become poisoned at any
    /// time. You should not trust a `false` value for program correctness
    /// without additional synchronization.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::thread;
    /// use std::sync::Arc;
    /// use jkcutils_rs::sync::SeqLock;
    /// 
    /// let seqlock = Arc::new(SeqLock::new(0));
    /// let c_seqlock = Arc::clone(&seqlock);
    ///
    /// let _ = thread::spawn(move || {
    ///     let _lock = c_seqlock.write().unwrap();
    ///     panic!(); // the seqlock gets poisoned
    /// }).join();
    /// assert_eq!(seqlock.is_poisoned(), true);
    /// ```
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poison.get()
    }

    /// Clear the poisoned state from a seqlock.
    /// 
    /// If the seqlock is poisoned, it will remain poisoned until this function is called. This
    /// allows recovering from a poisoned state and marking that it has recovered. For example, if
    /// the value is overwritten by a known-good value, then the seqlock can be marked as
    /// un-poisoned. Or possibly, the value could be inspected to determine if it is in a
    /// consistent state, and if so the poison is removed.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use std::thread;
    /// use std::sync::Arc;
    /// use jkcutils_rs::sync::SeqLock;
    /// 
    /// let seqlock = Arc::new(SeqLock::new(0));
    /// let c_seqlock = Arc::clone(&seqlock);
    ///
    /// let _ = thread::spawn(move || {
    ///     let _lock = c_seqlock.write().unwrap();
    ///     panic!(); // the seqlock gets poisoned
    /// }).join();
    ///
    /// assert_eq!(seqlock.is_poisoned(), true);
    /// let x = seqlock.write().unwrap_or_else(|mut e| {
    ///     **e.get_mut() = 1;
    ///     seqlock.clear_poison();
    ///     e.into_inner()
    /// });
    /// assert_eq!(seqlock.is_poisoned(), false);
    /// assert_eq!(*x, 1);
    /// ```
    #[inline]
    pub fn clear_poison(&self) {
        self.poison.clear();
    }

    /// Consumes this seqlock, returning the underlying data.
    /// 
    /// # Errors
    /// 
    /// If another user of this seqlock panicked while holding the seqlock, then
    /// this call will return an error instead.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SeqLock;
    /// 
    /// let seqlock = SeqLock::new(0);
    /// assert_eq!(seqlock.into_inner().unwrap(), 0);
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
    /// Since this call borrows the `SeqLock` mutably, no actual locking needs to
    /// take place -- the mutable borrow statically guarantees no locks exist.
    /// 
    /// Errors
    /// 
    /// If another user of this seqlock panicked while holding the seqlock, then
    /// this call will return an error instead.
    /// 
    /// # Examples
    /// 
    /// ```
    /// use jkcutils_rs::sync::SeqLock;
    /// 
    /// let mut seqlock = SeqLock::new(0);
    /// *seqlock.get_mut().unwrap() = 10;
    /// assert_eq!(*seqlock.write().unwrap(), 10);
    /// ```
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        let data = self.data.get_mut();

        poision::map_result(self.poison.borrow(), |()| data)
    }
}


impl<T> From<T> for SeqLock<T> {
    /// Crates a new seqlock in an unlocked state ready for use.
    /// This is equivalent to [`SeqLock::new`]
    fn from(t: T) -> Self {
        Self::new(t)
    }
}


impl<T: ?Sized + Default> Default for SeqLock<T> {
    /// Creates a `SeqLock<T>`, with the `Default` value for T.
    fn default() -> SeqLock<T> {
        SeqLock::new(Default::default())
    }
}


impl<T: ?Sized + fmt::Debug> fmt::Debug for SeqLock<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("SeqLock");
        match self.read() {
            Ok(guard) => {
                d.field("data", &&*guard);
            }
            Err(err) => {
                d.field("data", &&**err.get_ref());
            }
        }
        d.field("poisoned", &self.poison.get());
        d.finish_non_exhaustive()
    }
}


impl<'a, T: ?Sized> SeqLockWriteGuard<'a, T> {
    unsafe fn new(lock: &'a SeqLock<T>) -> LockResult<SeqLockWriteGuard<'a, T>> {
        lock.write_seqcount_begin();
        poision::map_result(lock.poison.guard(), |guard| Self { lock, poison: guard })
    }
}


impl<T: ?Sized> std::ops::Deref for SeqLockWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { & *self.lock.data.get() }
    }
}


impl<T: ?Sized> std::ops::DerefMut for SeqLockWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}


impl<T: ?Sized> Drop for SeqLockWriteGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.poison.done(&self.poison);
        self.lock.write_seqcount_end();
    }
}


impl<T: ?Sized + fmt::Debug> fmt::Debug for SeqLockWriteGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}


impl<T: ?Sized + fmt::Display> fmt::Display for SeqLockWriteGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}


impl<'a, T:? Sized> SeqLockReadGuard<'a, T> {
    unsafe fn new(lock: &'a SeqLock<T>) -> LockResult<SeqLockReadGuard<'a, T>> {
        poision::map_result(lock.poison.guard(), |guard| Self { lock, poison: guard })
    }
}


impl<T: ?Sized> std::ops::Deref for SeqLockReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        loop {
            let seq1 = self.lock.read_seqcount_begin();

            if seq1 % 2 != 0 {
                std::thread::yield_now();
                continue;
            }

            let data = unsafe { & *self.lock.data.get() };

            let seq2 = self.lock.read_seqcount_begin();

            if seq1 == seq2 {
                return data;
            }
        }
    }
}


impl<T: ?Sized> Drop for SeqLockReadGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        self.lock.poison.done(&self.poison);
    }
}


impl<T: ?Sized + fmt::Debug> fmt::Debug for SeqLockReadGuard<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}


impl<T: ?Sized + fmt::Display> fmt::Display for SeqLockReadGuard<'_, T> {
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
    fn test_seqlock() {
        let seqlock = SeqLock::new(Vec::new());

        std::thread::scope(|s| {
            s.spawn(|| {
                println!(">>> {:?}", seqlock.sequence);
                let value = seqlock.read().unwrap();
                assert!(value.as_slice() == vec![2, 3] || value.as_slice() == vec![]);
                println!(">>> {:?}", seqlock.sequence);
            });
            s.spawn(|| {
                let mut seqlock_guard = seqlock.write().unwrap();
                seqlock_guard.push(2);
                seqlock_guard.push(3);
                assert_eq!(seqlock.try_write().is_err(), true);
                drop(seqlock_guard);
            });
        });

        let value = seqlock.read().unwrap();
        assert!(value.as_slice() == vec![2, 3] || value.as_slice() == vec![]);
    }

    #[test]
    fn test_seqlock_poison() {
        let seqlock = Arc::new(SeqLock::new(0));
        let c_seqlock = seqlock.clone();

        println!(">>> {seqlock:?}");

        let _ = thread::spawn(move || {
            let _lock = c_seqlock.write().unwrap();
            panic!();
        }).join();

        assert_eq!(seqlock.is_poisoned(), true);
        assert_eq!(seqlock.read().is_err(), true);
        assert_eq!(seqlock.write().is_err(), true);

        println!(">>> {seqlock:?}");

        let x = seqlock.write().unwrap_or_else(|mut e| {
            **e.get_mut() = 1;
            seqlock.clear_poison();
            e.into_inner()
        });
        assert_eq!(seqlock.is_poisoned(), false);
        assert_eq!(*x, 1);
    }
}