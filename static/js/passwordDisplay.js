passwordInput = document.getElementById("password");
toggleButton = document.getElementById("togglePassword");

toggleButton.addEventListener("click", () => {
    const isPassword = passwordInput.type === "password";
    passwordInput.type = isPassword ? "text" : "password";
    toggleButton.textContent = isPassword ? "Hide" : "Show";
});