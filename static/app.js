const btn = document.getElementById("extract-btn")
const input = document.getElementById("input-text")
const listKeywords = document.getElementById("keywords-list")
const listKeyphrases = document.getElementById("keyphrases-list")
const similarBox = document.getElementById("similar-box")
const similarList = document.getElementById("similar-list")

function clearList(el) {
if (el) {
el.innerHTML = ""
}
}

function fillList(el, items) {
if (!el || !items) return
items.forEach(v => {
const li = document.createElement("li")
li.textContent = v
el.appendChild(li)
})
}

if (btn && input) {
btn.addEventListener("click", async () => {
const text = input.value

clearList(listKeywords)
clearList(listKeyphrases)
if (similarList) similarList.innerHTML = ""
if (similarBox) similarBox.style.display = "none"

if (!text.trim()) {
alert("Please paste some text first.")
return
}

const original = btn.textContent
btn.textContent = "Extracting..."
btn.disabled = true
btn.style.opacity = "0.6"

try {
const response = await fetch("/api/extract", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ text })
})
const data = await response.json()

if (data.error) {
alert(data.error)
return
}

fillList(listKeywords, data.keywords)
fillList(listKeyphrases, data.keyphrases)

if (similarBox && similarList) {
const items = data.similar_articles || []
similarList.innerHTML = ""
if (!items.length) {
similarBox.style.display = "none"
} else {
items.forEach(a => {
const item = document.createElement("div")
item.className = "similar-item"
const score = (a.similarity * 100).toFixed(1)
item.innerHTML =
"<div class=\"similar-score\">" + score + "% match</div>" +
"<div class=\"similar-snippet\">" + a.snippet + "</div>"
similarList.appendChild(item)
})
similarBox.style.display = "block"
}
}

} catch (e) {
alert("Network error while extracting.")
console.error(e)
} finally {
btn.textContent = original
btn.disabled = false
btn.style.opacity = "1"
}
})
}
