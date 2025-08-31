class Solution:
    # assumed to minimize the function x^2
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        ans = init

        for _ in range(iterations):
            deriv = 2 * ans
            ans -= deriv * learning_rate
        
        return round(ans, 5)
    
# Example usage:
ans = Solution()
print(ans.get_minimizer(0, 0.01, 5))
print(ans.get_minimizer(10, 0.01, 5))