// script.js

// Function to clear author filter
function clearAuthorFilter() {
    document.querySelectorAll('#author-filters input[type="radio"]').forEach(radio => {
        radio.checked = false;
    });
    filterBooks();
}

// In your existing script where you set up event listeners, add:
document.getElementById('clear-authors').addEventListener('click', clearAuthorFilter);

// Function to filter options based on search input
function filterOptions(inputId, containerId) {
    const searchText = document.getElementById(inputId).value.toLowerCase();
    const items = document.querySelectorAll(`#${containerId} label`);

    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        if (text.includes(searchText)) {
            item.previousElementSibling.style.display = '';
            item.style.display = '';
            item.nextElementSibling.style.display = '';
            item.nextElementSibling.nextElementSibling.style.display = '';
        } else {
            item.previousElementSibling.style.display = 'none';
            item.style.display = 'none';
            item.nextElementSibling.style.display = 'none';
            item.nextElementSibling.nextElementSibling.style.display = 'none';
        }
    });
}

// Event listeners for category and tag search inputs
document.getElementById('category-search').addEventListener('input', () => filterOptions('category-search', 'category-filters'));
document.getElementById('tag-search').addEventListener('input', () => filterOptions('tag-search', 'tag-filters'));

// Function to create checkbox or radio button filters with count
function createFilterWithCount(items, containerId, isRadio = false) {
    const checked = new Set([...document.querySelectorAll(`#${containerId} input:checked`)].map(x => x.id));
    items.sort((a, b) => {
        if(checked.has(a.name) && !checked.has(b.name)) {
            return -1;
        } else if (!checked.has(a.name) && checked.has(b.name)) {
            return 1;
        } else {
            return b.count - a.count;
        }
    });


    const container = document.getElementById(containerId);
    container.innerHTML = ''; // Clear existing content

    items.forEach(({name, count}) => {
        const input = document.createElement('input');
        input.type = isRadio ? 'radio' : 'checkbox';
        input.id = name;
        input.name = isRadio ? 'author-filter' : name; // Radio buttons need the same 'name' attribute
        input.value = name;
        input.checked = checked.has(name);

        const label = document.createElement('label');
        label.htmlFor = name;
        label.textContent = name;

        // Span for showing count
        const countSpan = document.createElement('span');
        countSpan.id = `count-${containerId}-${name}`;
        countSpan.style.marginLeft = '5px';
        countSpan.textContent = count;

        container.appendChild(input);
        container.appendChild(label);
        container.appendChild(countSpan);
        container.appendChild(document.createElement('br'));

        // Attach event listener to each input
        input.addEventListener('change', () => {
            filterBooks();
        });
    });
}

// Function to add filters
function addFilters(filteredBooks) {
    const categoryCounts = {};
    const tagCounts = {};
    const authorCounts = {};

    // Extracting unique categories, tags, and authors
    filteredBooks.forEach(book => {
        book.categories.forEach(category => {
            categoryCounts[category] = (categoryCounts[category] || 0) + 1;
        });
        book.tags.forEach(tag => {
            tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        });
        book.authors.forEach(author => {
            authorCounts[author] = (authorCounts[author] || 0) + 1;
        });
    });
    books.forEach(book => {
        book.authors.forEach(author => {
            authorCounts[author] = (authorCounts[author] || 0);
        });
    });

    const categories = Object.entries(categoryCounts).map(([name, count]) => ({ name, count }));
    const tags = Object.entries(tagCounts).map(([name, count]) => ({ name, count }));
    const authors = Object.entries(authorCounts).map(([name, count]) => ({ name, count }));

    // Creating checkbox filters with counts for categories, tags
    createFilterWithCount(categories, 'category-filters');
    createFilterWithCount(tags, 'tag-filters');

    // Creating radio button filters for authors
    createFilterWithCount(authors, 'author-filters', true);
}

// Function to filter books based on selected categories and tags
function filterBooks() {
    const selectedCategories = new Set();
    const selectedTags = new Set();
    const selectedAuthors = new Set();

    // Collecting selected categories
    document.querySelectorAll('#category-filters input:checked').forEach(checkbox => {
        selectedCategories.add(checkbox.value);
    });

    // Collecting selected tags
    document.querySelectorAll('#tag-filters input:checked').forEach(checkbox => {
        selectedTags.add(checkbox.value);
    });

    document.querySelectorAll('#author-filters input:checked').forEach(checkbox => {
        selectedAuthors.add(checkbox.value);
    });

    const filteredBooks = books.filter(book => {
        const categoryMatch = selectedCategories.size === 0 || Array.from(selectedCategories).every(category => book.categories.includes(category));
        const tagMatch = selectedTags.size === 0 || Array.from(selectedTags).every(tag => book.tags.includes(tag));
        const authorMatch = selectedAuthors.size === 0 || Array.from(selectedAuthors).every(author => book.authors.includes(author));
        return categoryMatch && tagMatch && authorMatch;
    });

    displayBooks(filteredBooks);
    addFilters(filteredBooks);
}

// Modified displayBooks function to accept an array of books
function displayBooks(filteredBooks) {
    const booksContainer = document.getElementById('books');
    booksContainer.innerHTML = '';

    const max = Math.min(50, filteredBooks.length);
    for (let ind = 0; ind < max; ind++) {
        const book = filteredBooks[ind]
        const bookElement = document.createElement('a');
        bookElement.target = "_blank";
        bookElement.href = book.href;
        bookElement.className = 'book';
        bookElement.innerHTML = `
            <img src="${book.coverUrl}" alt="${book.name}">
            <span>${book.name}</span>
        `;
        booksContainer.appendChild(bookElement);
    }
}


// Initial setup
filterBooks();
