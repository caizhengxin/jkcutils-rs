/// Finds subsequence of a sequence
#[inline(always)]
pub fn find_subsequence<T>(haystack: &[T], needle: &[T]) -> Option<usize>
where
    for<'a> &'a [T]: PartialEq,
{
    let needle_len = needle.len();
    haystack
        .windows(needle_len)
        .position(|window| window == needle)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_subsequence() {
        let input = [0x00, 0x01, 0x02, 0x00];

        assert_eq!(find_subsequence(&input, &[0x00]), Some(0));
        assert_eq!(find_subsequence(&input, &[0x00, 0x01]), Some(0));
        assert_eq!(find_subsequence(&input, &[0x01, 0x03]), None);
        assert_eq!(find_subsequence(&input, &[0x01, 0x02]), Some(1));
        assert_eq!(find_subsequence(&input, &[0x02, 0x00]), Some(2));
    }
}