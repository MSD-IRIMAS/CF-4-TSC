// Function to load paper's data from json
$(document).ready(function() {
	$.getJSON("static/jsons/paper_data.json", function(data){
		$("#page-title").append(data.title);
		$("#paper-title").append(data.title);
		for (var i=0; i<data.authors.length; i++){
			var string_author='<span class=\"author-block\">';
			string_author = string_author + '<a href=\"' + data.authors[i].link + '\">' + data.authors[i].name + '</a><sup>' + data.authors[i].institution_id + '</sup>';
			if (i<data.authors.length-1){
				string_author = string_author + ', </span>';
			}else{
				string_author = string_author + '</span>';
			}
			$("#paper-authors").append(string_author);
		}
		for (var i=0; i<data.institutions.length; i++){
			$("#paper-institutions").append('<span class=\"author-block\"><sup>'+ String(i+1) + '</sup>' + data.institutions[i] + '</span>');
			if (i<data.institutions.length-1){
				$("#paper-institutions").append('<br/>');
			}
		}
		if (data.hasOwnProperty('pdf_link')) {
			$("#paper-links").append(' <span class=\"link-block\"><a href=\"' + data.pdf_link + '\" class=\"external-link button is-normal is-rounded is-dark\"><span class=\"icon\"><i class=\"fas fa-file-pdf\"></i></span><span>Paper</span></a></span> ');
		}
		if (data.hasOwnProperty('doi_link')) {
			$("#paper-links").append(' <span class=\"link-block\"><a href=\"' + data.doi_link + '\" class=\"external-link button is-normal is-rounded is-dark\"><span class=\"icon\"><i class=\"fas fa-link\"></i></span><span>DOI</span></a></span> ');
		}
		if (data.hasOwnProperty('slides_link')) {
			$("#paper-links").append(' <span class=\"link-block\"><a href=\"' + data.slides_link + '\" class=\"external-link button is-normal is-rounded is-dark\"><span class=\"icon\"><i class=\"fab fa-slideshare\"></i></span><span>Slides</span></a></span> ');
		}
		if (data.hasOwnProperty('code_link')) {
			$("#paper-links").append(' <span class=\"link-block\"><a href=\"' + data.code_link + '\" class=\"external-link button is-normal is-rounded is-dark\"><span class=\"icon\"><i class=\"fab fa-github\"></i></span><span>Code</span></a></span> ');
		}
		$("#img-overview").attr("src", data.image_overview);
		$("#caption-overview").append(data.caption_overview);
		$("#abstract").append(data.abstract);
		if (data.hasOwnProperty('acknowledgment')) {
			$("#acknowledgment").append(data.acknowledgment);
		}
		if (data.hasOwnProperty('webpage_code')) {
			$("#webpage-link").attr("href", data.webpage_code)
		}
		//load specific sections (results) from external htmls
		if (data.hasOwnProperty('external_htmls')) {
			for (var i=0; i<data.external_htmls.length; i++){
				$('#specific-parts').load(data.external_htmls[i]);
			}
		}
	});
});


// Function to load bitex
$(document).ready(function(){
	$.get('static/bibtex/bitex.bib', function(data) {
		res = bibtexParse.toJSON(data)[0];
		if (res.entryType == "inproceedings"){
			$("#bibtex-info").append('@' + res.entryType + '{' + res.citationKey + ',\n author = {' + res.entryTags.author + '},\n  title = {' + res.entryTags.title + '},\n  booktitle = {' + res.entryTags.booktitle + '},\n  city = {' + res.entryTags.city + '},\n  country = {' + res.entryTags.country + '},\n  pages = {' + res.entryTags.pages + '},\n  year = {'+ res.entryTags.year + '},\n  organization = {' + res.entryTags.organization + '}\n}');
		}else{
			$("#bibtex-info").append('@' + res.entryType + '{' + res.citationKey + ',\n author = {' + res.entryTags.author + '},\n  title = {' + res.entryTags.title + '},\n  journal = {' + res.entryTags.journal + '},\n  volume = {' + res.entryTags.volume + '},\n  number = {' + res.entryTags.number + '},\n  pages = {' + res.entryTags.pages + '},\n  year = {'+ res.entryTags.year + '},\n  publisher = {' + res.entryTags.publisher + '}\n}');
		}
}, 'text');
});

