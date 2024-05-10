const BOOKS_LIMIT = 50;

const categorySearch = document.getElementById('category-search');
categorySearch.addEventListener('input', () => filterOptions('category-search', 'category-filters'));
const tagSearch = document.getElementById('tag-search');
tagSearch.addEventListener('input', () => filterOptions('tag-search', 'tag-filters'));

function clearAuthorFilter() {
    document.querySelectorAll('#author-filters input[type="radio"]:checked').forEach(radio => {
        radio.checked = false;
    });
}
function clearFilters() {
    document.querySelectorAll('input:checked').forEach(radio => {
        radio.checked = false;
    });
}

document.getElementById('clear-authors').addEventListener('click', () => {
    clearAuthorFilter();
    filterBooks();
});

document.addEventListener('click', ({target}) => {
    if (target && target.className === 'book-author' && target.dataset.author) {
        clearFilters();
        const filter = document.getElementById(target.dataset.author);
        filter.checked = true;
        filter.scrollIntoView(false);
        filterBooks();
    }
});
document.addEventListener('change', ({target}) => {
    if (target && target.tagName === 'INPUT' && (target.type === 'checkbox' || target.type === 'radio')) {
        categorySearch.value = '';
        tagSearch.value = '';
        filterBooks();
    } else if (target && target.tagName === 'SELECT') {
        const queryParams = new URLSearchParams(location.search);
        queryParams.set('sorting', target.value)
        history.pushState({queryParams: queryParams.toString()}, '', `?${queryParams.toString()}`);

        sortBooks(target.value);
        filterBooks(false);
    }
});

window.addEventListener('popstate', (event) => {
    clearFilters();

    if (event.state && event.state.queryParams) {
        initFromQueryString(event.state.queryParams);
    }
});

const randomBooksButton = document.getElementById('randomBooks');
randomBooksButton.addEventListener('click', function() {
    filterBooks(false, true);
});

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

// Function to create checkbox or radio button filters with count
function createFilterWithCount(items, containerId, isRadio = false) {
    const checked = new Set([...document.querySelectorAll(`#${containerId} input:checked`)].map(x => x.id));
    if (isRadio) {
        items.sort((a, b) => {
            return b.count - a.count;
        });
    } else {
        items.sort((a, b) => {
            if (checked.has(a.name) && !checked.has(b.name)) {
                return -1;
            } else if (!checked.has(a.name) && checked.has(b.name)) {
                return 1;
            } else {
                return b.count - a.count;
            }
        });
    }

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
        label.appendChild(countSpan);
        container.appendChild(document.createElement('br'));
    });

    if(isRadio) {
        document.querySelector(`#${containerId} input:checked`)?.scrollIntoView();
    }
}

function createAuthors() {
    const authorCounts = {};
    books.forEach(book => {
        book.authors.forEach(author => {
            authorCounts[author] = authorCounts[author] ?? {
                name: author,
                count: 0,
                total: 0,
            };
            authorCounts[author].total = authorCounts[author].count = authorCounts[author].count + 1;
        });
    });
    const authors = Object.values(authorCounts);
    authors.sort((a, b) => {
        return b.count - a.count;
    });
    return [authorCounts, authors];
}

// Function to add filters
function addFilters(filteredBooks, filterBooksWithoutAuthor) {
    for (const author of authors) {
        author.count = 0;
    }

    const categoryCounts = {};
    const tagCounts = {};

    // Extracting unique categories, tags, and authors
    filteredBooks.forEach(book => {
        book.categories.forEach(category => {
            categoryCounts[category] = (categoryCounts[category] || 0) + 1;
        });
        book.tags.forEach(tag => {
            tagCounts[tag] = (tagCounts[tag] || 0) + 1;
        });
    });
    filterBooksWithoutAuthor.forEach(book => {
        book.authors.forEach(author => {
            authorCounts[author].count = authorCounts[author].count + 1;
        });
    });

    const categories = Object.entries(categoryCounts).map(([name, count]) => ({name, count}));
    const tags = Object.entries(tagCounts).map(([name, count]) => ({name, count}));

    // Creating checkbox filters with counts for categories, tags
    createFilterWithCount(categories, 'category-filters');
    createFilterWithCount(tags, 'tag-filters');

    // Creating radio button filters for authors
    createFilterWithCount(authors, 'author-filters', true);
}

// Function to filter books based on selected categories and tags
function filterBooks(updateUrl = true, randomize = false) {
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

    if (updateUrl) {
        // Constructing query string
        const queryParams = new URLSearchParams(location.search);
        queryParams.delete('selected');
        selectedCategories.forEach(category => queryParams.append('selected', category));
        selectedTags.forEach(tag => queryParams.append('selected', tag));
        selectedAuthors.forEach(author => queryParams.append('selected', author));

        // Update the URL with the new query string
        history.pushState({queryParams: queryParams.toString()}, '', `?${queryParams.toString()}`);
    }

    const categories = Array.from(selectedCategories);
    const tags = Array.from(selectedTags);
    const authors = Array.from(selectedAuthors);
    const filterBooksWithoutAuthor = (categories.length || tags.length) ? books.filter(book => {
        const categoryMatch = categories.length === 0 || categories.every(category => book.categories.includes(category));
        const tagMatch = tags.length === 0 || tags.every(tag => book.tags.includes(tag));
        return categoryMatch && tagMatch;
    }) : books;
    const filteredBooks = (authors.length) ? filterBooksWithoutAuthor.filter(book => {
        return  authors.length === 0 || authors.every(author => book.authors.includes(author));
    }) : filterBooksWithoutAuthor;

    displayBooks(randomize ? random(filteredBooks) : filteredBooks);
    addFilters(filteredBooks, filterBooksWithoutAuthor);

    randomBooksButton.disabled = filteredBooks.length <= BOOKS_LIMIT;
}
function random(filteredBooks, limit = BOOKS_LIMIT) {
    if (filteredBooks.length <= limit) {
        return filteredBooks;
    }

    const set = new Set();
    while(set.size < limit) {
        console.count('while');
        set.add(Math.floor(Math.random() * filteredBooks.length));
    }

    return [...set].map(x => filteredBooks[x]);
}

// Modified displayBooks function to accept an array of books
function displayBooks(filteredBooks, limit = BOOKS_LIMIT) {
    const booksContainer = document.getElementById('books');
    booksContainer.innerHTML = '';

    let shown = 0;
    function showNext() {
        const from = shown;
        shown = Math.min(filteredBooks.length, shown + limit);

        for (let ind = from; ind < shown; ind++) {
            const book = filteredBooks[ind];
            const author = book.authors[0];
            const authorCount = authorCounts[author]?.total;
            const bookElement = document.createElement('div');
            const coverUrl = `images/${images[book.id]?.image}`;
            const offset = images[book.id]?.index * 200;
            bookElement.className = 'book';
            bookElement.innerHTML = `
            <a target="_blank" href="${book.href}"><img src="${coverUrl}" alt="${book.name}" style="object-fit: none; object-position: 0 -${offset}px; width: 200px; height: 200px"></a>
            <div>
                <span class="name">${book.name}</span>
                <span class="book-author" data-author="${author}">${authorCount}</span>
            </div>
        `;
            booksContainer.appendChild(bookElement);
        }

        if (shown !== filteredBooks.length) {
            const bookElement = document.createElement('div');
            bookElement.className = 'book show-more';
            bookElement.innerHTML = `Show more`;
            bookElement.addEventListener('click', () => {
                bookElement.remove();
                showNext();
            });
            booksContainer.appendChild(bookElement);
        }
    }
    showNext();
}

function initFromQueryString(queryString) {
    const queryParams = new URLSearchParams(queryString);

    // Check the boxes that match the query parameters
    queryParams.getAll('selected').forEach(value => {
        const input = document.getElementById(value);
        if (input) input.checked = true;
    });

    const sorting = queryParams.get('sorting') ?? 'name';
    document.getElementById('sorting').value = sorting;

    sortBooks(sorting)
    filterBooks(false);
}

function sortBooks(term) {
    switch (term) {
        case 'name': {
            books.sort((a, b) => a.name.localeCompare(b.name));
            break;
        }
        case '-name': {
            books.sort((a, b) => b.name.localeCompare(a.name));
            break;
        }
        case 'pages': {
            books.sort((a, b) => a.pages - b.pages);
            break;
        }
        case '-pages': {
            books.sort((a, b) => b.pages - a.pages);
            break;
        }
        case 'score': {
            books.sort((a, b) => a.score - b.score);
            break;
        }
        case '-score': {
            books.sort((a, b) => b.score - a.score);
            break;
        }
        case 'views': {
            books.sort((a, b) => a.views - b.views);
            break;
        }
        case '-views': {
            books.sort((a, b) => b.views - a.views);
            break;
        }
        case 'coverScore': {
            books.sort((a, b) => a.coverScore - b.coverScore);
            break;
        }
        case '-coverScore': {
            books.sort((a, b) => b.coverScore - a.coverScore);
            break;
        }
    }
}

const [authorCounts, authors] = createAuthors();
addFilters(books, books);
initFromQueryString(location.search);
